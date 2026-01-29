"""
GLNORM Error Handling Module for GL-FOUND-X-003.

This module provides comprehensive error handling for the GreenLang Normalizer
(GLNORM) component, including error codes, response models, factory functions,
and custom exception classes.

The error system follows these principles:
- Structured error codes with clear taxonomy (E1xx-E9xx)
- Consistent response models following RFC 7807
- Actionable suggestions for error resolution
- Full audit trail support for compliance
- Exception hierarchy matching error categories

Quick Start:
    >>> from gl_normalizer_core.errors import (
    ...     GLNORMErrorCode,
    ...     GLNORMErrorFactory,
    ...     GLNORMUnitParsingError,
    ... )
    >>>
    >>> # Create an error response
    >>> factory = GLNORMErrorFactory()
    >>> error = factory.create_error(
    ...     code=GLNORMErrorCode.E100_UNIT_PARSE_FAILED,
    ...     message="Failed to parse unit 'xyz'"
    ... )
    >>>
    >>> # Raise an exception
    >>> raise GLNORMUnitParsingError(
    ...     code=GLNORMErrorCode.E100_UNIT_PARSE_FAILED,
    ...     message="Failed to parse unit",
    ...     input_unit="xyz"
    ... )

Error Categories:
    - E1xx: Unit Parsing Errors - Lexical/syntactic issues with unit strings
    - E2xx: Dimension Errors - Dimensional analysis failures
    - E3xx: Conversion Errors - Unit conversion failures
    - E4xx: Entity Resolution Errors - Reference data matching issues
    - E5xx: Vocabulary Errors - Vocabulary registry issues
    - E6xx: Audit Errors - Audit trail failures
    - E9xx: System/Limit Errors - Infrastructure and resource issues

Module Components:
    - codes: Error code enumeration with metadata
    - response: Pydantic models for error responses
    - factory: Error factory with suggestion generation
    - exceptions: Custom exception classes

Example Usage:

    Creating Error Responses::

        from gl_normalizer_core.errors import create_error, GLNORMErrorCode

        # Simple error
        error = create_error(
            GLNORMErrorCode.E101_UNKNOWN_UNIT,
            "Unknown unit: 'xyz'"
        )

        # Error with details
        error = create_error(
            GLNORMErrorCode.E305_GWP_VERSION_MISSING,
            "GWP version required",
            details={"source_unit": "kg-ch4", "target_unit": "kg-co2e"},
            operation="convert_unit"
        )

    Raising Exceptions::

        from gl_normalizer_core.errors import (
            GLNORMUnknownUnitError,
            GLNORMGWPVersionMissingError,
        )

        # Unknown unit
        raise GLNORMUnknownUnitError(
            input_unit="xyz",
            vocabularies_searched=["gl-vocab-units-v2"]
        )

        # Missing GWP version
        raise GLNORMGWPVersionMissingError(
            source_unit="kg-ch4"
        )

    Handling Exceptions::

        from gl_normalizer_core.errors import (
            GLNORMException,
            GLNORMErrorFactory,
        )

        factory = GLNORMErrorFactory()

        try:
            # ... operation that may fail
            pass
        except GLNORMException as e:
            # Convert to response
            response = factory.create_error(
                code=e.code,
                message=e.message,
                details=e.details
            )
            return response

    Validation Errors::

        from gl_normalizer_core.errors import create_validation_error

        error = create_validation_error([
            {"field": "source_unit", "message": "Required", "value": None},
            {"field": "value", "message": "Must be positive", "value": -5}
        ])
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# =============================================================================
# Error Codes
# =============================================================================

from .codes import (
    # Main enum
    GLNORMErrorCode,
    # Category metadata
    CATEGORY_NAMES,
    CATEGORY_DESCRIPTIONS,
    # Error classifications
    RECOVERABLE_ERRORS,
    HUMAN_REVIEW_REQUIRED,
    CRITICAL_ERRORS,
    WARNING_ERRORS,
    INFO_ERRORS,
    # HTTP status mapping
    ERROR_HTTP_STATUS,
    # Utility functions
    get_error_code_by_value,
    get_codes_by_category,
    get_all_codes,
)

# =============================================================================
# Response Models
# =============================================================================

from .response import (
    # Context models
    ErrorContext,
    ErrorCandidate,
    ErrorSuggestion,
    # Error detail models
    ErrorDetail,
    ValidationErrorItem,
    # Response models
    GLNORMErrorResponse,
    GLNORMBatchErrorResponse,
    GLNORMValidationErrorResponse,
    # Audit model
    AuditableError,
)

# =============================================================================
# Factory
# =============================================================================

from .factory import (
    # Factory class
    GLNORMErrorFactory,
    # Factory management
    get_error_factory,
    set_error_factory,
    # Convenience functions
    create_error,
    create_validation_error,
)

# =============================================================================
# Exceptions
# =============================================================================

from .exceptions import (
    # Base exception
    GLNORMException,
    # E1xx: Unit Parsing Errors
    GLNORMUnitParsingError,
    GLNORMUnknownUnitError,
    GLNORMAmbiguousUnitError,
    # E2xx: Dimension Errors
    GLNORMDimensionError,
    GLNORMDimensionMismatchError,
    # E3xx: Conversion Errors
    GLNORMConversionError,
    GLNORMMissingReferenceConditionsError,
    GLNORMGWPVersionMissingError,
    # E4xx: Entity Resolution Errors
    GLNORMEntityResolutionError,
    GLNORMReferenceNotFoundError,
    GLNORMLowConfidenceMatchError,
    # E5xx: Vocabulary Errors
    GLNORMVocabularyError,
    # E6xx: Audit Errors
    GLNORMAuditError,
    GLNORMAuditIntegrityViolationError,
    # E9xx: System Errors
    GLNORMSystemError,
    GLNORMLimitExceededError,
    GLNORMTimeoutError,
    # Exception utilities
    EXCEPTION_REGISTRY,
    get_exception_class,
    raise_for_code,
)

# =============================================================================
# Legacy Error Types (Backward Compatibility)
# =============================================================================


class ErrorDetails(BaseModel):
    """Structured error details for API responses (legacy)."""

    code: str = Field(..., description="Error code for programmatic handling")
    message: str = Field(..., description="Human-readable error message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    hint: Optional[str] = Field(None, description="Suggestion for resolution")


class NormalizerError(Exception):
    """
    Base exception for all normalizer errors (legacy).

    Note: New code should use GLNORMException instead.

    Attributes:
        message: Human-readable error message
        code: Machine-readable error code
        details: Additional context about the error
    """

    code: str = "NORMALIZER_ERROR"

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        hint: Optional[str] = None,
    ) -> None:
        """Initialize NormalizerError."""
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        self.details = details or {}
        self.hint = hint

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses."""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "hint": self.hint,
        }

    def to_model(self) -> ErrorDetails:
        """Convert error to Pydantic model."""
        return ErrorDetails(
            code=self.code,
            message=self.message,
            details=self.details,
            hint=self.hint,
        )


class ParseError(NormalizerError):
    """
    Error raised when parsing a quantity string fails (legacy).

    Note: New code should use GLNORMUnitParsingError instead.
    """

    code: str = "PARSE_ERROR"

    def __init__(
        self,
        message: str,
        input_value: Optional[str] = None,
        position: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ParseError."""
        details = kwargs.pop("details", {})
        if input_value:
            details["input_value"] = input_value
        if position is not None:
            details["position"] = position
        super().__init__(message, details=details, **kwargs)
        self.input_value = input_value
        self.position = position


class ConversionError(NormalizerError):
    """
    Error raised when unit conversion fails (legacy).

    Note: New code should use GLNORMConversionError instead.
    """

    code: str = "CONVERSION_ERROR"

    def __init__(
        self,
        message: str,
        source_unit: Optional[str] = None,
        target_unit: Optional[str] = None,
        reason: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ConversionError."""
        details = kwargs.pop("details", {})
        if source_unit:
            details["source_unit"] = source_unit
        if target_unit:
            details["target_unit"] = target_unit
        if reason:
            details["reason"] = reason
        super().__init__(message, details=details, **kwargs)
        self.source_unit = source_unit
        self.target_unit = target_unit
        self.reason = reason


class DimensionMismatchError(ConversionError):
    """
    Error raised when units have incompatible dimensions (legacy).

    Note: New code should use GLNORMDimensionMismatchError instead.
    """

    code: str = "DIMENSION_MISMATCH"

    def __init__(
        self,
        message: str,
        source_dimension: Optional[str] = None,
        target_dimension: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize DimensionMismatchError."""
        details = kwargs.pop("details", {})
        if source_dimension:
            details["source_dimension"] = source_dimension
        if target_dimension:
            details["target_dimension"] = target_dimension
        hint = kwargs.pop("hint", None) or (
            f"Cannot convert between {source_dimension} and {target_dimension}. "
            "Units must have the same physical dimension."
        )
        super().__init__(message, details=details, hint=hint, **kwargs)
        self.source_dimension = source_dimension
        self.target_dimension = target_dimension


class ResolutionError(NormalizerError):
    """
    Error raised when reference resolution fails (legacy).

    Note: New code should use GLNORMEntityResolutionError instead.
    """

    code: str = "RESOLUTION_ERROR"

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        vocabulary: Optional[str] = None,
        candidates: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ResolutionError."""
        details = kwargs.pop("details", {})
        if query:
            details["query"] = query
        if vocabulary:
            details["vocabulary"] = vocabulary
        if candidates:
            details["candidates"] = candidates
        super().__init__(message, details=details, **kwargs)
        self.query = query
        self.vocabulary = vocabulary
        self.candidates = candidates or []


class VocabularyError(NormalizerError):
    """
    Error raised when vocabulary operations fail (legacy).

    Note: New code should use GLNORMVocabularyError instead.
    """

    code: str = "VOCABULARY_ERROR"

    def __init__(
        self,
        message: str,
        vocabulary_id: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize VocabularyError."""
        details = kwargs.pop("details", {})
        if vocabulary_id:
            details["vocabulary_id"] = vocabulary_id
        if version:
            details["version"] = version
        super().__init__(message, details=details, **kwargs)
        self.vocabulary_id = vocabulary_id
        self.version = version


class PolicyViolationError(NormalizerError):
    """
    Error raised when a conversion violates policy rules (legacy).
    """

    code: str = "POLICY_VIOLATION"

    def __init__(
        self,
        message: str,
        policy_id: Optional[str] = None,
        rule_id: Optional[str] = None,
        compliance_profile: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize PolicyViolationError."""
        details = kwargs.pop("details", {})
        if policy_id:
            details["policy_id"] = policy_id
        if rule_id:
            details["rule_id"] = rule_id
        if compliance_profile:
            details["compliance_profile"] = compliance_profile
        super().__init__(message, details=details, **kwargs)
        self.policy_id = policy_id
        self.rule_id = rule_id
        self.compliance_profile = compliance_profile


class AuditError(NormalizerError):
    """Error raised when audit operations fail (legacy)."""

    code: str = "AUDIT_ERROR"


class CacheError(NormalizerError):
    """Error raised when cache operations fail (legacy)."""

    code: str = "CACHE_ERROR"


class ConfigurationError(NormalizerError):
    """Error raised when configuration is invalid (legacy)."""

    code: str = "CONFIGURATION_ERROR"


# =============================================================================
# Module Metadata
# =============================================================================

__version__ = "1.0.0"
__author__ = "GreenLang Team"

__all__ = [
    # Version
    "__version__",
    # ==========================================================================
    # GLNORM Error Taxonomy (New)
    # ==========================================================================
    # Error Codes
    "GLNORMErrorCode",
    "CATEGORY_NAMES",
    "CATEGORY_DESCRIPTIONS",
    "RECOVERABLE_ERRORS",
    "HUMAN_REVIEW_REQUIRED",
    "CRITICAL_ERRORS",
    "WARNING_ERRORS",
    "INFO_ERRORS",
    "ERROR_HTTP_STATUS",
    "get_error_code_by_value",
    "get_codes_by_category",
    "get_all_codes",
    # Response Models
    "ErrorContext",
    "ErrorCandidate",
    "ErrorSuggestion",
    "ErrorDetail",
    "ValidationErrorItem",
    "GLNORMErrorResponse",
    "GLNORMBatchErrorResponse",
    "GLNORMValidationErrorResponse",
    "AuditableError",
    # Factory
    "GLNORMErrorFactory",
    "get_error_factory",
    "set_error_factory",
    "create_error",
    "create_validation_error",
    # Base Exception
    "GLNORMException",
    # E1xx Exceptions
    "GLNORMUnitParsingError",
    "GLNORMUnknownUnitError",
    "GLNORMAmbiguousUnitError",
    # E2xx Exceptions
    "GLNORMDimensionError",
    "GLNORMDimensionMismatchError",
    # E3xx Exceptions
    "GLNORMConversionError",
    "GLNORMMissingReferenceConditionsError",
    "GLNORMGWPVersionMissingError",
    # E4xx Exceptions
    "GLNORMEntityResolutionError",
    "GLNORMReferenceNotFoundError",
    "GLNORMLowConfidenceMatchError",
    # E5xx Exceptions
    "GLNORMVocabularyError",
    # E6xx Exceptions
    "GLNORMAuditError",
    "GLNORMAuditIntegrityViolationError",
    # E9xx Exceptions
    "GLNORMSystemError",
    "GLNORMLimitExceededError",
    "GLNORMTimeoutError",
    # Exception Utilities
    "EXCEPTION_REGISTRY",
    "get_exception_class",
    "raise_for_code",
    # ==========================================================================
    # Legacy Error Types (Backward Compatibility)
    # ==========================================================================
    "ErrorDetails",
    "NormalizerError",
    "ParseError",
    "ConversionError",
    "DimensionMismatchError",
    "ResolutionError",
    "VocabularyError",
    "PolicyViolationError",
    "AuditError",
    "CacheError",
    "ConfigurationError",
]
