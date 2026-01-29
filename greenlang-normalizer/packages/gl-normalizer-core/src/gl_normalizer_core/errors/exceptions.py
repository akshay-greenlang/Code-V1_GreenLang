"""
GLNORM Custom Exception Classes for GL-FOUND-X-003.

This module defines custom exception classes for the GreenLang Normalizer
component. Each exception class corresponds to an error category and includes
structured error information for consistent error handling.

Exception Hierarchy:
    GLNORMException (base)
    |-- GLNORMUnitParsingError (E1xx)
    |-- GLNORMDimensionError (E2xx)
    |-- GLNORMConversionError (E3xx)
    |-- GLNORMEntityResolutionError (E4xx)
    |-- GLNORMVocabularyError (E5xx)
    |-- GLNORMAuditError (E6xx)
    |-- GLNORMSystemError (E9xx)

Example:
    >>> from gl_normalizer_core.errors.exceptions import GLNORMUnitParsingError
    >>> from gl_normalizer_core.errors.codes import GLNORMErrorCode
    >>> raise GLNORMUnitParsingError(
    ...     code=GLNORMErrorCode.E100_UNIT_PARSE_FAILED,
    ...     message="Failed to parse unit 'xyz'",
    ...     details={"input": "xyz", "position": 0}
    ... )
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union
from uuid import UUID, uuid4

from .codes import GLNORMErrorCode, CATEGORY_NAMES

logger = logging.getLogger(__name__)


class GLNORMException(Exception):
    """
    Base exception class for all GLNORM errors.

    This base class provides structured error information including
    error code, message, details, and context for consistent error
    handling across the normalizer component.

    Attributes:
        code: GLNORM error code
        message: Human-readable error message
        details: Error-specific details dictionary
        request_id: Request identifier for tracking
        timestamp: When the exception was raised
        candidates: Candidate values for ambiguous errors
        is_recoverable: Whether error is potentially recoverable
        requires_human_review: Whether human review is recommended

    Example:
        >>> try:
        ...     raise GLNORMException(
        ...         code=GLNORMErrorCode.E904_INTERNAL_ERROR,
        ...         message="Unexpected condition"
        ...     )
        ... except GLNORMException as e:
        ...     print(f"Error {e.code.value}: {e.message}")
    """

    def __init__(
        self,
        code: GLNORMErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[UUID] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize GLNORMException.

        Args:
            code: GLNORM error code
            message: Human-readable error message
            details: Error-specific details
            request_id: Request identifier
            candidates: Candidate values for ambiguous errors
            cause: Original exception that caused this error
        """
        self.code = code
        self.message = message
        self.details = details or {}
        self.request_id = request_id or uuid4()
        self.timestamp = datetime.utcnow()
        self.candidates = candidates
        self.cause = cause

        # Derived properties
        self.is_recoverable = code.is_recoverable
        self.requires_human_review = code.requires_human_review
        self.severity = code.severity
        self.http_status = code.http_status
        self.category = code.category_name

        # Build full message
        full_message = f"[{code.value}] {message}"
        super().__init__(full_message)

        # Log the exception
        self._log_exception()

    def _log_exception(self) -> None:
        """Log the exception with appropriate severity."""
        log_data = {
            "error_code": self.code.value,
            "message": self.message,
            "request_id": str(self.request_id),
            "details": self.details,
        }

        if self.severity == "CRITICAL":
            logger.critical(f"GLNORM Error: {log_data}", exc_info=self.cause)
        elif self.severity == "ERROR":
            logger.error(f"GLNORM Error: {log_data}", exc_info=self.cause)
        elif self.severity == "WARNING":
            logger.warning(f"GLNORM Warning: {log_data}")
        else:
            logger.info(f"GLNORM Info: {log_data}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary representation.

        Returns:
            Dictionary with all exception details

        Example:
            >>> e = GLNORMException(code, message)
            >>> d = e.to_dict()
            >>> assert d["code"] == code.value
        """
        return {
            "code": self.code.value,
            "message": self.message,
            "category": self.category,
            "severity": self.severity,
            "details": self.details,
            "request_id": str(self.request_id),
            "timestamp": self.timestamp.isoformat(),
            "is_recoverable": self.is_recoverable,
            "requires_human_review": self.requires_human_review,
            "http_status": self.http_status,
            "candidates": self.candidates,
        }

    def __str__(self) -> str:
        """Return string representation."""
        return f"[{self.code.value}] {self.message}"

    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"{self.__class__.__name__}("
            f"code={self.code.value!r}, "
            f"message={self.message!r}, "
            f"request_id={self.request_id!r})"
        )


# =============================================================================
# E1xx: Unit Parsing Errors
# =============================================================================


class GLNORMUnitParsingError(GLNORMException):
    """
    Exception for unit parsing errors (E1xx).

    Raised when the unit parser fails to tokenize, parse, or interpret
    a unit string. Includes information about the parse position and
    expected tokens.

    Attributes:
        input_unit: The original input unit string
        position: Character position where parsing failed (0-indexed)
        expected: What was expected at the failure position

    Example:
        >>> raise GLNORMUnitParsingError(
        ...     code=GLNORMErrorCode.E100_UNIT_PARSE_FAILED,
        ...     message="Unexpected character at position 5",
        ...     input_unit="kgCO2/kwh",
        ...     position=5
        ... )
    """

    def __init__(
        self,
        code: GLNORMErrorCode,
        message: str,
        input_unit: Optional[str] = None,
        position: Optional[int] = None,
        expected: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize GLNORMUnitParsingError.

        Args:
            code: Error code (must be E1xx)
            message: Error message
            input_unit: The original input unit string
            position: Character position of error
            expected: What was expected
            **kwargs: Additional arguments for base class
        """
        self.input_unit = input_unit
        self.position = position
        self.expected = expected

        details = kwargs.pop("details", {})
        details.update({
            "input_unit": input_unit,
            "position": position,
            "expected": expected,
        })
        # Remove None values
        details = {k: v for k, v in details.items() if v is not None}

        super().__init__(code=code, message=message, details=details, **kwargs)


class GLNORMUnknownUnitError(GLNORMUnitParsingError):
    """
    Exception for unknown unit symbols (E101).

    Raised when a unit symbol is not found in any registered vocabulary.

    Example:
        >>> raise GLNORMUnknownUnitError(
        ...     input_unit="xyz",
        ...     vocabularies_searched=["gl-vocab-units-v2"]
        ... )
    """

    def __init__(
        self,
        input_unit: str,
        vocabularies_searched: Optional[List[str]] = None,
        similar_units: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """
        Initialize GLNORMUnknownUnitError.

        Args:
            input_unit: The unknown unit symbol
            vocabularies_searched: Vocabularies that were searched
            similar_units: Similar units that might be intended
            **kwargs: Additional arguments
        """
        message = f"Unknown unit: '{input_unit}'"
        if vocabularies_searched:
            message += f" (searched: {', '.join(vocabularies_searched)})"

        details = kwargs.pop("details", {})
        details.update({
            "vocabularies_searched": vocabularies_searched,
            "similar_units": similar_units,
        })

        super().__init__(
            code=GLNORMErrorCode.E101_UNKNOWN_UNIT,
            message=message,
            input_unit=input_unit,
            details=details,
            **kwargs,
        )


class GLNORMAmbiguousUnitError(GLNORMUnitParsingError):
    """
    Exception for ambiguous unit interpretations (E104).

    Raised when a unit string matches multiple interpretations
    without a clear winner.

    Example:
        >>> raise GLNORMAmbiguousUnitError(
        ...     input_unit="t",
        ...     candidates=[
        ...         {"value": "tonne", "label": "Metric tonne"},
        ...         {"value": "ton", "label": "US short ton"}
        ...     ]
        ... )
    """

    def __init__(
        self,
        input_unit: str,
        candidates: List[Dict[str, Any]],
        **kwargs: Any,
    ):
        """
        Initialize GLNORMAmbiguousUnitError.

        Args:
            input_unit: The ambiguous unit string
            candidates: List of candidate interpretations
            **kwargs: Additional arguments
        """
        message = f"Ambiguous unit: '{input_unit}' matches {len(candidates)} interpretations"

        super().__init__(
            code=GLNORMErrorCode.E104_AMBIGUOUS_UNIT,
            message=message,
            input_unit=input_unit,
            candidates=candidates,
            **kwargs,
        )


# =============================================================================
# E2xx: Dimension Errors
# =============================================================================


class GLNORMDimensionError(GLNORMException):
    """
    Exception for dimension analysis errors (E2xx).

    Raised when dimensional analysis fails, including dimension
    mismatches, unknown dimensions, and incompatibilities.

    Attributes:
        source_dimension: Dimension of source unit
        target_dimension: Dimension of target unit (if applicable)

    Example:
        >>> raise GLNORMDimensionError(
        ...     code=GLNORMErrorCode.E200_DIMENSION_MISMATCH,
        ...     message="Cannot convert length to mass",
        ...     source_dimension="L",
        ...     target_dimension="M"
        ... )
    """

    def __init__(
        self,
        code: GLNORMErrorCode,
        message: str,
        source_dimension: Optional[str] = None,
        target_dimension: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize GLNORMDimensionError.

        Args:
            code: Error code (must be E2xx)
            message: Error message
            source_dimension: Dimension of source unit
            target_dimension: Dimension of target unit
            **kwargs: Additional arguments
        """
        self.source_dimension = source_dimension
        self.target_dimension = target_dimension

        details = kwargs.pop("details", {})
        details.update({
            "source_dimension": source_dimension,
            "target_dimension": target_dimension,
        })
        details = {k: v for k, v in details.items() if v is not None}

        super().__init__(code=code, message=message, details=details, **kwargs)


class GLNORMDimensionMismatchError(GLNORMDimensionError):
    """
    Exception for dimension mismatches (E200).

    Raised when attempting to convert between units with different dimensions.

    Example:
        >>> raise GLNORMDimensionMismatchError(
        ...     source_unit="kg",
        ...     target_unit="m",
        ...     source_dimension="M",
        ...     target_dimension="L"
        ... )
    """

    def __init__(
        self,
        source_unit: str,
        target_unit: str,
        source_dimension: str,
        target_dimension: str,
        **kwargs: Any,
    ):
        """
        Initialize GLNORMDimensionMismatchError.

        Args:
            source_unit: Source unit symbol
            target_unit: Target unit symbol
            source_dimension: Source dimension
            target_dimension: Target dimension
            **kwargs: Additional arguments
        """
        message = (
            f"Dimension mismatch: cannot convert '{source_unit}' ({source_dimension}) "
            f"to '{target_unit}' ({target_dimension})"
        )

        details = kwargs.pop("details", {})
        details.update({
            "source_unit": source_unit,
            "target_unit": target_unit,
        })

        super().__init__(
            code=GLNORMErrorCode.E200_DIMENSION_MISMATCH,
            message=message,
            source_dimension=source_dimension,
            target_dimension=target_dimension,
            details=details,
            **kwargs,
        )


# =============================================================================
# E3xx: Conversion Errors
# =============================================================================


class GLNORMConversionError(GLNORMException):
    """
    Exception for unit conversion errors (E3xx).

    Raised when unit conversion fails due to missing conversion paths,
    reference conditions, or calculation errors.

    Attributes:
        source_unit: Source unit symbol
        target_unit: Target unit symbol
        value: Value being converted (if applicable)

    Example:
        >>> raise GLNORMConversionError(
        ...     code=GLNORMErrorCode.E300_CONVERSION_NOT_SUPPORTED,
        ...     message="No conversion path found",
        ...     source_unit="custom-unit",
        ...     target_unit="kg"
        ... )
    """

    def __init__(
        self,
        code: GLNORMErrorCode,
        message: str,
        source_unit: Optional[str] = None,
        target_unit: Optional[str] = None,
        value: Optional[float] = None,
        **kwargs: Any,
    ):
        """
        Initialize GLNORMConversionError.

        Args:
            code: Error code (must be E3xx)
            message: Error message
            source_unit: Source unit
            target_unit: Target unit
            value: Value being converted
            **kwargs: Additional arguments
        """
        self.source_unit = source_unit
        self.target_unit = target_unit
        self.value = value

        details = kwargs.pop("details", {})
        details.update({
            "source_unit": source_unit,
            "target_unit": target_unit,
            "value": value,
        })
        details = {k: v for k, v in details.items() if v is not None}

        super().__init__(code=code, message=message, details=details, **kwargs)


class GLNORMMissingReferenceConditionsError(GLNORMConversionError):
    """
    Exception for missing reference conditions (E301).

    Raised when a conversion requires reference conditions (temperature,
    pressure, etc.) but none were provided.

    Example:
        >>> raise GLNORMMissingReferenceConditionsError(
        ...     source_unit="m3-gas",
        ...     target_unit="kg",
        ...     required_conditions=["temperature_k", "pressure_pa"]
        ... )
    """

    def __init__(
        self,
        source_unit: str,
        target_unit: str,
        required_conditions: List[str],
        **kwargs: Any,
    ):
        """
        Initialize GLNORMMissingReferenceConditionsError.

        Args:
            source_unit: Source unit
            target_unit: Target unit
            required_conditions: List of required condition fields
            **kwargs: Additional arguments
        """
        message = (
            f"Reference conditions required for '{source_unit}' to '{target_unit}': "
            f"{', '.join(required_conditions)}"
        )

        details = kwargs.pop("details", {})
        details["required_conditions"] = required_conditions

        super().__init__(
            code=GLNORMErrorCode.E301_MISSING_REFERENCE_CONDITIONS,
            message=message,
            source_unit=source_unit,
            target_unit=target_unit,
            details=details,
            **kwargs,
        )


class GLNORMGWPVersionMissingError(GLNORMConversionError):
    """
    Exception for missing GWP version (E305).

    Raised when converting to CO2 equivalents without specifying
    the GWP version.

    Example:
        >>> raise GLNORMGWPVersionMissingError(
        ...     source_unit="kg-ch4",
        ...     target_unit="kg-co2e"
        ... )
    """

    def __init__(
        self,
        source_unit: str,
        target_unit: str = "kg-co2e",
        **kwargs: Any,
    ):
        """
        Initialize GLNORMGWPVersionMissingError.

        Args:
            source_unit: Source greenhouse gas unit
            target_unit: Target CO2e unit
            **kwargs: Additional arguments
        """
        message = f"GWP version required for '{source_unit}' to '{target_unit}' conversion"

        # Add GWP version candidates
        candidates = [
            {"value": "AR6", "label": "IPCC AR6 (2021) - Recommended"},
            {"value": "AR5", "label": "IPCC AR5 (2014)"},
            {"value": "AR4", "label": "IPCC AR4 (2007)"},
        ]

        super().__init__(
            code=GLNORMErrorCode.E305_GWP_VERSION_MISSING,
            message=message,
            source_unit=source_unit,
            target_unit=target_unit,
            candidates=candidates,
            **kwargs,
        )


# =============================================================================
# E4xx: Entity Resolution Errors
# =============================================================================


class GLNORMEntityResolutionError(GLNORMException):
    """
    Exception for entity resolution errors (E4xx).

    Raised when entity resolution fails, including not found,
    ambiguous matches, and low confidence matches.

    Attributes:
        entity_type: Type of entity being resolved
        query: The resolution query
        vocabulary_id: Vocabulary being searched

    Example:
        >>> raise GLNORMEntityResolutionError(
        ...     code=GLNORMErrorCode.E400_REFERENCE_NOT_FOUND,
        ...     message="Entity not found",
        ...     entity_type="emission_factor",
        ...     query="natural-gas-combustion"
        ... )
    """

    def __init__(
        self,
        code: GLNORMErrorCode,
        message: str,
        entity_type: Optional[str] = None,
        query: Optional[str] = None,
        vocabulary_id: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize GLNORMEntityResolutionError.

        Args:
            code: Error code (must be E4xx)
            message: Error message
            entity_type: Type of entity
            query: Resolution query
            vocabulary_id: Vocabulary ID
            **kwargs: Additional arguments
        """
        self.entity_type = entity_type
        self.query = query
        self.vocabulary_id = vocabulary_id

        details = kwargs.pop("details", {})
        details.update({
            "entity_type": entity_type,
            "query": query,
            "vocabulary_id": vocabulary_id,
        })
        details = {k: v for k, v in details.items() if v is not None}

        super().__init__(code=code, message=message, details=details, **kwargs)


class GLNORMReferenceNotFoundError(GLNORMEntityResolutionError):
    """
    Exception for entity not found (E400).

    Raised when an entity reference cannot be found in any vocabulary.

    Example:
        >>> raise GLNORMReferenceNotFoundError(
        ...     entity_id="unknown-entity-id",
        ...     entity_type="emission_factor"
        ... )
    """

    def __init__(
        self,
        entity_id: str,
        entity_type: str,
        vocabularies_searched: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """
        Initialize GLNORMReferenceNotFoundError.

        Args:
            entity_id: The entity ID that was not found
            entity_type: Type of entity
            vocabularies_searched: Vocabularies that were searched
            **kwargs: Additional arguments
        """
        message = f"Entity not found: '{entity_id}' (type: {entity_type})"
        if vocabularies_searched:
            message += f" in vocabularies: {', '.join(vocabularies_searched)}"

        details = kwargs.pop("details", {})
        details.update({
            "entity_id": entity_id,
            "vocabularies_searched": vocabularies_searched,
        })

        super().__init__(
            code=GLNORMErrorCode.E400_REFERENCE_NOT_FOUND,
            message=message,
            entity_type=entity_type,
            query=entity_id,
            details=details,
            **kwargs,
        )


class GLNORMLowConfidenceMatchError(GLNORMEntityResolutionError):
    """
    Exception for low confidence entity matches (E403).

    Raised when the best entity match has confidence below the threshold.

    Example:
        >>> raise GLNORMLowConfidenceMatchError(
        ...     query="natrual gas combuston",
        ...     best_match="natural-gas-combustion",
        ...     confidence=0.65,
        ...     threshold=0.80
        ... )
    """

    def __init__(
        self,
        query: str,
        best_match: str,
        confidence: float,
        threshold: float = 0.80,
        entity_type: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize GLNORMLowConfidenceMatchError.

        Args:
            query: The resolution query
            best_match: The best matching entity
            confidence: Match confidence (0.0-1.0)
            threshold: Required confidence threshold
            entity_type: Type of entity
            **kwargs: Additional arguments
        """
        message = (
            f"Low confidence match for '{query}': best match '{best_match}' "
            f"has confidence {confidence:.2f} (threshold: {threshold:.2f})"
        )

        details = kwargs.pop("details", {})
        details.update({
            "best_match": best_match,
            "confidence": confidence,
            "threshold": threshold,
        })

        candidates = [
            {
                "value": best_match,
                "label": f"Best match (confidence: {confidence:.2f})",
                "confidence": confidence,
            }
        ]

        super().__init__(
            code=GLNORMErrorCode.E403_LOW_CONFIDENCE_MATCH,
            message=message,
            entity_type=entity_type,
            query=query,
            candidates=candidates,
            details=details,
            **kwargs,
        )


# =============================================================================
# E5xx: Vocabulary Errors
# =============================================================================


class GLNORMVocabularyError(GLNORMException):
    """
    Exception for vocabulary errors (E5xx).

    Raised when vocabulary loading, validation, or access fails.

    Attributes:
        vocabulary_id: Vocabulary identifier
        vocabulary_version: Vocabulary version

    Example:
        >>> raise GLNORMVocabularyError(
        ...     code=GLNORMErrorCode.E501_VOCABULARY_LOAD_FAILED,
        ...     message="Failed to load vocabulary",
        ...     vocabulary_id="gl-vocab-units-v2"
        ... )
    """

    def __init__(
        self,
        code: GLNORMErrorCode,
        message: str,
        vocabulary_id: Optional[str] = None,
        vocabulary_version: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize GLNORMVocabularyError.

        Args:
            code: Error code (must be E5xx)
            message: Error message
            vocabulary_id: Vocabulary identifier
            vocabulary_version: Vocabulary version
            **kwargs: Additional arguments
        """
        self.vocabulary_id = vocabulary_id
        self.vocabulary_version = vocabulary_version

        details = kwargs.pop("details", {})
        details.update({
            "vocabulary_id": vocabulary_id,
            "vocabulary_version": vocabulary_version,
        })
        details = {k: v for k, v in details.items() if v is not None}

        super().__init__(code=code, message=message, details=details, **kwargs)


# =============================================================================
# E6xx: Audit Errors
# =============================================================================


class GLNORMAuditError(GLNORMException):
    """
    Exception for audit trail errors (E6xx).

    Raised when audit logging, verification, or replay fails.

    Attributes:
        audit_id: Audit record identifier
        operation: Operation that was being audited

    Example:
        >>> raise GLNORMAuditError(
        ...     code=GLNORMErrorCode.E600_AUDIT_WRITE_FAILED,
        ...     message="Failed to write audit record",
        ...     audit_id="audit-123"
        ... )
    """

    def __init__(
        self,
        code: GLNORMErrorCode,
        message: str,
        audit_id: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize GLNORMAuditError.

        Args:
            code: Error code (must be E6xx)
            message: Error message
            audit_id: Audit record identifier
            operation: Operation being audited
            **kwargs: Additional arguments
        """
        self.audit_id = audit_id
        self.operation = operation

        details = kwargs.pop("details", {})
        details.update({
            "audit_id": audit_id,
            "operation": operation,
        })
        details = {k: v for k, v in details.items() if v is not None}

        super().__init__(code=code, message=message, details=details, **kwargs)


class GLNORMAuditIntegrityViolationError(GLNORMAuditError):
    """
    Exception for audit integrity violations (E602).

    Raised when audit record tampering is detected. This is a
    CRITICAL severity error requiring immediate investigation.

    Example:
        >>> raise GLNORMAuditIntegrityViolationError(
        ...     audit_id="audit-123",
        ...     expected_hash="abc123...",
        ...     actual_hash="def456..."
        ... )
    """

    def __init__(
        self,
        audit_id: str,
        expected_hash: str,
        actual_hash: str,
        **kwargs: Any,
    ):
        """
        Initialize GLNORMAuditIntegrityViolationError.

        Args:
            audit_id: Audit record identifier
            expected_hash: Expected integrity hash
            actual_hash: Actual computed hash
            **kwargs: Additional arguments
        """
        message = (
            f"Audit integrity violation detected for record '{audit_id}': "
            f"hash mismatch (expected: {expected_hash[:16]}..., actual: {actual_hash[:16]}...)"
        )

        details = kwargs.pop("details", {})
        details.update({
            "expected_hash": expected_hash,
            "actual_hash": actual_hash,
        })

        super().__init__(
            code=GLNORMErrorCode.E602_AUDIT_INTEGRITY_VIOLATION,
            message=message,
            audit_id=audit_id,
            details=details,
            **kwargs,
        )


# =============================================================================
# E9xx: System Errors
# =============================================================================


class GLNORMSystemError(GLNORMException):
    """
    Exception for system-level errors (E9xx).

    Raised for infrastructure issues including limits, timeouts,
    resource exhaustion, and internal errors.

    Attributes:
        service: Service that failed (if applicable)
        limit_name: Name of limit exceeded (if applicable)

    Example:
        >>> raise GLNORMSystemError(
        ...     code=GLNORMErrorCode.E901_TIMEOUT,
        ...     message="Operation timed out after 30s"
        ... )
    """

    def __init__(
        self,
        code: GLNORMErrorCode,
        message: str,
        service: Optional[str] = None,
        limit_name: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize GLNORMSystemError.

        Args:
            code: Error code (must be E9xx)
            message: Error message
            service: Service that failed
            limit_name: Name of limit exceeded
            **kwargs: Additional arguments
        """
        self.service = service
        self.limit_name = limit_name

        details = kwargs.pop("details", {})
        details.update({
            "service": service,
            "limit_name": limit_name,
        })
        details = {k: v for k, v in details.items() if v is not None}

        super().__init__(code=code, message=message, details=details, **kwargs)


class GLNORMLimitExceededError(GLNORMSystemError):
    """
    Exception for limit exceeded errors (E900).

    Raised when an operation exceeds configured limits.

    Example:
        >>> raise GLNORMLimitExceededError(
        ...     limit_name="batch_size",
        ...     current_value=10000,
        ...     max_value=1000
        ... )
    """

    def __init__(
        self,
        limit_name: str,
        current_value: int,
        max_value: int,
        **kwargs: Any,
    ):
        """
        Initialize GLNORMLimitExceededError.

        Args:
            limit_name: Name of the limit
            current_value: Current value that exceeded limit
            max_value: Maximum allowed value
            **kwargs: Additional arguments
        """
        message = f"Limit exceeded: {limit_name} ({current_value} > {max_value})"

        details = kwargs.pop("details", {})
        details.update({
            "current_value": current_value,
            "max_value": max_value,
        })

        super().__init__(
            code=GLNORMErrorCode.E900_LIMIT_EXCEEDED,
            message=message,
            limit_name=limit_name,
            details=details,
            **kwargs,
        )


class GLNORMTimeoutError(GLNORMSystemError):
    """
    Exception for timeout errors (E901).

    Raised when an operation times out.

    Example:
        >>> raise GLNORMTimeoutError(
        ...     operation="vocabulary_load",
        ...     timeout_seconds=30
        ... )
    """

    def __init__(
        self,
        operation: str,
        timeout_seconds: float,
        **kwargs: Any,
    ):
        """
        Initialize GLNORMTimeoutError.

        Args:
            operation: Operation that timed out
            timeout_seconds: Timeout duration in seconds
            **kwargs: Additional arguments
        """
        message = f"Operation '{operation}' timed out after {timeout_seconds}s"

        details = kwargs.pop("details", {})
        details.update({
            "operation": operation,
            "timeout_seconds": timeout_seconds,
        })

        super().__init__(
            code=GLNORMErrorCode.E901_TIMEOUT,
            message=message,
            details=details,
            **kwargs,
        )


# =============================================================================
# Exception Registry
# =============================================================================

EXCEPTION_REGISTRY: Dict[int, Type[GLNORMException]] = {
    1: GLNORMUnitParsingError,
    2: GLNORMDimensionError,
    3: GLNORMConversionError,
    4: GLNORMEntityResolutionError,
    5: GLNORMVocabularyError,
    6: GLNORMAuditError,
    9: GLNORMSystemError,
}


def get_exception_class(code: GLNORMErrorCode) -> Type[GLNORMException]:
    """
    Get the appropriate exception class for an error code.

    Args:
        code: GLNORM error code

    Returns:
        Exception class for the error category

    Example:
        >>> cls = get_exception_class(GLNORMErrorCode.E100_UNIT_PARSE_FAILED)
        >>> assert cls == GLNORMUnitParsingError
    """
    return EXCEPTION_REGISTRY.get(code.category, GLNORMException)


def raise_for_code(
    code: GLNORMErrorCode,
    message: str,
    **kwargs: Any,
) -> None:
    """
    Raise the appropriate exception for an error code.

    Args:
        code: GLNORM error code
        message: Error message
        **kwargs: Additional exception arguments

    Raises:
        Appropriate GLNORMException subclass

    Example:
        >>> raise_for_code(
        ...     GLNORMErrorCode.E100_UNIT_PARSE_FAILED,
        ...     "Failed to parse unit"
        ... )
    """
    exception_class = get_exception_class(code)
    raise exception_class(code=code, message=message, **kwargs)
