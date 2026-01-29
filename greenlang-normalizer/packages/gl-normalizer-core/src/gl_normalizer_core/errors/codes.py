"""
GLNORM Error Code Taxonomy for GL-FOUND-X-003.

This module defines the complete error code enumeration for the GreenLang
Normalizer (GLNORM) component. Error codes follow a structured taxonomy:

- E1xx: Unit Parsing Errors (lexical/syntactic issues with unit strings)
- E2xx: Dimension Errors (dimensional analysis failures)
- E3xx: Conversion Errors (unit conversion failures)
- E4xx: Entity Resolution Errors (reference data matching issues)
- E5xx: Vocabulary Errors (vocabulary registry issues)
- E6xx: Audit Errors (audit trail failures)
- E9xx: System/Limit Errors (infrastructure and resource issues)

Each error code follows the format: GLNORM-E{category}{sequence}

Example:
    >>> from gl_normalizer_core.errors.codes import GLNORMErrorCode
    >>> code = GLNORMErrorCode.E100_UNIT_PARSE_FAILED
    >>> print(code.value)  # GLNORM-E100
    >>> print(code.category)  # 1
    >>> print(code.is_recoverable)  # False
"""

from enum import Enum
from typing import Dict, Optional


class GLNORMErrorCode(str, Enum):
    """
    Complete enumeration of GLNORM error codes.

    Error codes are structured as GLNORM-E{NNN} where:
    - GLNORM: Component identifier
    - E: Error indicator
    - NNN: Three-digit code (first digit = category)

    Attributes:
        value: The string representation of the error code (e.g., "GLNORM-E100")

    Example:
        >>> code = GLNORMErrorCode.E100_UNIT_PARSE_FAILED
        >>> assert code.value == "GLNORM-E100"
        >>> assert code.category == 1
    """

    # =========================================================================
    # E1xx: Unit Parsing Errors
    # =========================================================================
    # These errors occur during lexical/syntactic analysis of unit strings

    E100_UNIT_PARSE_FAILED = "GLNORM-E100"
    """General unit parsing failure - unable to tokenize or parse unit string."""

    E101_UNKNOWN_UNIT = "GLNORM-E101"
    """Unit symbol/name not found in any registered vocabulary."""

    E102_INVALID_PREFIX = "GLNORM-E102"
    """SI or binary prefix not recognized (e.g., 'mega', 'kilo')."""

    E103_INVALID_EXPONENT = "GLNORM-E103"
    """Invalid exponent in unit expression (e.g., 'm^abc')."""

    E104_AMBIGUOUS_UNIT = "GLNORM-E104"
    """Unit string matches multiple interpretations without clear winner."""

    E105_UNSUPPORTED_COMPOUND = "GLNORM-E105"
    """Compound unit structure not supported (e.g., nested fractions)."""

    E106_LOCALE_PARSE_ERROR = "GLNORM-E106"
    """Locale-specific parsing failed (e.g., decimal separator issues)."""

    # =========================================================================
    # E2xx: Dimension Errors
    # =========================================================================
    # These errors occur during dimensional analysis

    E200_DIMENSION_MISMATCH = "GLNORM-E200"
    """Source and target dimensions do not match for conversion."""

    E201_DIMENSION_UNKNOWN = "GLNORM-E201"
    """Dimension of unit cannot be determined."""

    E202_DIMENSION_INCOMPATIBLE = "GLNORM-E202"
    """Dimensions are fundamentally incompatible (e.g., length vs mass)."""

    E203_DIMENSIONLESS_EXPECTED = "GLNORM-E203"
    """Operation requires dimensionless quantity but got dimensioned."""

    E204_DIMENSION_EXPECTED = "GLNORM-E204"
    """Operation requires dimensioned quantity but got dimensionless."""

    # =========================================================================
    # E3xx: Conversion Errors
    # =========================================================================
    # These errors occur during unit conversion operations

    E300_CONVERSION_NOT_SUPPORTED = "GLNORM-E300"
    """Conversion path not available between source and target units."""

    E301_MISSING_REFERENCE_CONDITIONS = "GLNORM-E301"
    """Reference conditions required but not provided (e.g., STP for gases)."""

    E302_INVALID_REFERENCE_CONDITIONS = "GLNORM-E302"
    """Reference conditions provided are invalid or out of range."""

    E303_CONVERSION_FACTOR_MISSING = "GLNORM-E303"
    """Conversion factor not found in vocabulary registry."""

    E304_PRECISION_OVERFLOW = "GLNORM-E304"
    """Numeric precision exceeded during conversion calculation."""

    E305_GWP_VERSION_MISSING = "GLNORM-E305"
    """GWP version required for CO2e conversion but not specified."""

    E306_BASIS_MISSING = "GLNORM-E306"
    """Basis (e.g., HHV/LHV, wet/dry) required but not provided."""

    E307_CONVERSION_FACTOR_DEPRECATED = "GLNORM-E307"
    """Conversion factor is deprecated; newer version available."""

    # =========================================================================
    # E4xx: Entity Resolution Errors
    # =========================================================================
    # These errors occur during entity resolution and reference data matching

    E400_REFERENCE_NOT_FOUND = "GLNORM-E400"
    """Referenced entity not found in any vocabulary."""

    E401_REFERENCE_AMBIGUOUS = "GLNORM-E401"
    """Multiple candidate entities match with similar confidence."""

    E402_ENTITY_DEPRECATED = "GLNORM-E402"
    """Matched entity is deprecated; replacement may be available."""

    E403_LOW_CONFIDENCE_MATCH = "GLNORM-E403"
    """Best match confidence below threshold (requires human review)."""

    E404_VOCABULARY_NOT_FOUND = "GLNORM-E404"
    """Specified vocabulary identifier not found in registry."""

    E405_ENTITY_TYPE_MISMATCH = "GLNORM-E405"
    """Entity found but type does not match expected (e.g., activity vs emission)."""

    E406_ALIAS_COLLISION = "GLNORM-E406"
    """Alias maps to multiple canonical entities in different vocabularies."""

    E407_LLM_CANDIDATE_ONLY = "GLNORM-E407"
    """Only LLM-suggested candidates available; no deterministic match."""

    # =========================================================================
    # E5xx: Vocabulary Errors
    # =========================================================================
    # These errors occur during vocabulary loading and validation

    E500_VOCABULARY_VERSION_MISMATCH = "GLNORM-E500"
    """Vocabulary version incompatible with normalizer version."""

    E501_VOCABULARY_LOAD_FAILED = "GLNORM-E501"
    """Failed to load vocabulary from storage."""

    E502_VOCABULARY_CORRUPTED = "GLNORM-E502"
    """Vocabulary data integrity check failed."""

    E503_VOCABULARY_EXPIRED = "GLNORM-E503"
    """Vocabulary validity period has expired."""

    E504_GOVERNANCE_REQUIRED = "GLNORM-E504"
    """Vocabulary change requires governance approval."""

    # =========================================================================
    # E6xx: Audit Errors
    # =========================================================================
    # These errors occur during audit trail operations

    E600_AUDIT_WRITE_FAILED = "GLNORM-E600"
    """Failed to write audit record to store."""

    E601_AUDIT_STORE_UNAVAILABLE = "GLNORM-E601"
    """Audit store is unavailable or unreachable."""

    E602_AUDIT_INTEGRITY_VIOLATION = "GLNORM-E602"
    """Audit record integrity check failed (tampering detected)."""

    E603_REPLAY_MISMATCH = "GLNORM-E603"
    """Audit replay produced different result than original."""

    # =========================================================================
    # E9xx: System/Limit Errors
    # =========================================================================
    # These errors relate to system limits and infrastructure

    E900_LIMIT_EXCEEDED = "GLNORM-E900"
    """Operation limit exceeded (batch size, rate, etc.)."""

    E901_TIMEOUT = "GLNORM-E901"
    """Operation timed out."""

    E902_RESOURCE_EXHAUSTED = "GLNORM-E902"
    """System resource exhausted (memory, connections, etc.)."""

    E903_SERVICE_UNAVAILABLE = "GLNORM-E903"
    """Dependent service unavailable."""

    E904_INTERNAL_ERROR = "GLNORM-E904"
    """Internal error; unexpected condition."""

    # =========================================================================
    # Instance Properties
    # =========================================================================

    @property
    def category(self) -> int:
        """
        Extract the category (first digit) from the error code.

        Returns:
            int: Category number (1-9)

        Example:
            >>> GLNORMErrorCode.E100_UNIT_PARSE_FAILED.category
            1
            >>> GLNORMErrorCode.E904_INTERNAL_ERROR.category
            9
        """
        # Extract numeric part after 'GLNORM-E'
        numeric_part = self.value.replace("GLNORM-E", "")
        return int(numeric_part[0])

    @property
    def sequence(self) -> int:
        """
        Extract the sequence number (last two digits) from the error code.

        Returns:
            int: Sequence number (00-99)

        Example:
            >>> GLNORMErrorCode.E100_UNIT_PARSE_FAILED.sequence
            0
            >>> GLNORMErrorCode.E307_CONVERSION_FACTOR_DEPRECATED.sequence
            7
        """
        numeric_part = self.value.replace("GLNORM-E", "")
        return int(numeric_part[1:])

    @property
    def category_name(self) -> str:
        """
        Get the human-readable category name.

        Returns:
            str: Category name

        Example:
            >>> GLNORMErrorCode.E100_UNIT_PARSE_FAILED.category_name
            'Unit Parsing'
        """
        return CATEGORY_NAMES.get(self.category, "Unknown")

    @property
    def is_recoverable(self) -> bool:
        """
        Check if this error type is potentially recoverable.

        Returns:
            bool: True if error may be recoverable with intervention

        Example:
            >>> GLNORMErrorCode.E301_MISSING_REFERENCE_CONDITIONS.is_recoverable
            True
            >>> GLNORMErrorCode.E904_INTERNAL_ERROR.is_recoverable
            False
        """
        return self in RECOVERABLE_ERRORS

    @property
    def requires_human_review(self) -> bool:
        """
        Check if this error requires human review.

        Returns:
            bool: True if human review is recommended

        Example:
            >>> GLNORMErrorCode.E403_LOW_CONFIDENCE_MATCH.requires_human_review
            True
        """
        return self in HUMAN_REVIEW_REQUIRED

    @property
    def severity(self) -> str:
        """
        Get the default severity level for this error.

        Returns:
            str: One of 'CRITICAL', 'ERROR', 'WARNING', 'INFO'

        Example:
            >>> GLNORMErrorCode.E904_INTERNAL_ERROR.severity
            'CRITICAL'
            >>> GLNORMErrorCode.E307_CONVERSION_FACTOR_DEPRECATED.severity
            'WARNING'
        """
        if self in CRITICAL_ERRORS:
            return "CRITICAL"
        elif self in WARNING_ERRORS:
            return "WARNING"
        elif self in INFO_ERRORS:
            return "INFO"
        return "ERROR"

    @property
    def http_status(self) -> int:
        """
        Get the appropriate HTTP status code for this error.

        Returns:
            int: HTTP status code (400, 404, 422, 500, 503)

        Example:
            >>> GLNORMErrorCode.E100_UNIT_PARSE_FAILED.http_status
            400
            >>> GLNORMErrorCode.E400_REFERENCE_NOT_FOUND.http_status
            404
        """
        return ERROR_HTTP_STATUS.get(self, 500)


# =============================================================================
# Category Metadata
# =============================================================================

CATEGORY_NAMES: Dict[int, str] = {
    1: "Unit Parsing",
    2: "Dimension",
    3: "Conversion",
    4: "Entity Resolution",
    5: "Vocabulary",
    6: "Audit",
    9: "System",
}

CATEGORY_DESCRIPTIONS: Dict[int, str] = {
    1: "Errors during lexical and syntactic analysis of unit strings",
    2: "Errors during dimensional analysis of units",
    3: "Errors during unit conversion operations",
    4: "Errors during entity resolution and reference data matching",
    5: "Errors during vocabulary loading and validation",
    6: "Errors during audit trail operations",
    9: "System-level errors including limits and infrastructure",
}

# =============================================================================
# Error Classifications
# =============================================================================

RECOVERABLE_ERRORS: frozenset = frozenset({
    GLNORMErrorCode.E104_AMBIGUOUS_UNIT,
    GLNORMErrorCode.E301_MISSING_REFERENCE_CONDITIONS,
    GLNORMErrorCode.E302_INVALID_REFERENCE_CONDITIONS,
    GLNORMErrorCode.E305_GWP_VERSION_MISSING,
    GLNORMErrorCode.E306_BASIS_MISSING,
    GLNORMErrorCode.E307_CONVERSION_FACTOR_DEPRECATED,
    GLNORMErrorCode.E401_REFERENCE_AMBIGUOUS,
    GLNORMErrorCode.E402_ENTITY_DEPRECATED,
    GLNORMErrorCode.E403_LOW_CONFIDENCE_MATCH,
    GLNORMErrorCode.E407_LLM_CANDIDATE_ONLY,
    GLNORMErrorCode.E503_VOCABULARY_EXPIRED,
    GLNORMErrorCode.E504_GOVERNANCE_REQUIRED,
    GLNORMErrorCode.E900_LIMIT_EXCEEDED,
    GLNORMErrorCode.E901_TIMEOUT,
})

HUMAN_REVIEW_REQUIRED: frozenset = frozenset({
    GLNORMErrorCode.E104_AMBIGUOUS_UNIT,
    GLNORMErrorCode.E401_REFERENCE_AMBIGUOUS,
    GLNORMErrorCode.E403_LOW_CONFIDENCE_MATCH,
    GLNORMErrorCode.E406_ALIAS_COLLISION,
    GLNORMErrorCode.E407_LLM_CANDIDATE_ONLY,
    GLNORMErrorCode.E504_GOVERNANCE_REQUIRED,
    GLNORMErrorCode.E602_AUDIT_INTEGRITY_VIOLATION,
    GLNORMErrorCode.E603_REPLAY_MISMATCH,
})

CRITICAL_ERRORS: frozenset = frozenset({
    GLNORMErrorCode.E502_VOCABULARY_CORRUPTED,
    GLNORMErrorCode.E602_AUDIT_INTEGRITY_VIOLATION,
    GLNORMErrorCode.E603_REPLAY_MISMATCH,
    GLNORMErrorCode.E902_RESOURCE_EXHAUSTED,
    GLNORMErrorCode.E904_INTERNAL_ERROR,
})

WARNING_ERRORS: frozenset = frozenset({
    GLNORMErrorCode.E307_CONVERSION_FACTOR_DEPRECATED,
    GLNORMErrorCode.E402_ENTITY_DEPRECATED,
    GLNORMErrorCode.E403_LOW_CONFIDENCE_MATCH,
    GLNORMErrorCode.E407_LLM_CANDIDATE_ONLY,
    GLNORMErrorCode.E503_VOCABULARY_EXPIRED,
})

INFO_ERRORS: frozenset = frozenset({
    # Currently no info-level errors; placeholder for future use
})

# =============================================================================
# HTTP Status Mapping
# =============================================================================

ERROR_HTTP_STATUS: Dict[GLNORMErrorCode, int] = {
    # 400 Bad Request - Client input errors
    GLNORMErrorCode.E100_UNIT_PARSE_FAILED: 400,
    GLNORMErrorCode.E101_UNKNOWN_UNIT: 400,
    GLNORMErrorCode.E102_INVALID_PREFIX: 400,
    GLNORMErrorCode.E103_INVALID_EXPONENT: 400,
    GLNORMErrorCode.E104_AMBIGUOUS_UNIT: 400,
    GLNORMErrorCode.E105_UNSUPPORTED_COMPOUND: 400,
    GLNORMErrorCode.E106_LOCALE_PARSE_ERROR: 400,
    GLNORMErrorCode.E200_DIMENSION_MISMATCH: 400,
    GLNORMErrorCode.E201_DIMENSION_UNKNOWN: 400,
    GLNORMErrorCode.E202_DIMENSION_INCOMPATIBLE: 400,
    GLNORMErrorCode.E203_DIMENSIONLESS_EXPECTED: 400,
    GLNORMErrorCode.E204_DIMENSION_EXPECTED: 400,
    GLNORMErrorCode.E302_INVALID_REFERENCE_CONDITIONS: 400,
    GLNORMErrorCode.E405_ENTITY_TYPE_MISMATCH: 400,

    # 404 Not Found - Resource not found
    GLNORMErrorCode.E303_CONVERSION_FACTOR_MISSING: 404,
    GLNORMErrorCode.E400_REFERENCE_NOT_FOUND: 404,
    GLNORMErrorCode.E404_VOCABULARY_NOT_FOUND: 404,

    # 409 Conflict - Ambiguity/collision
    GLNORMErrorCode.E401_REFERENCE_AMBIGUOUS: 409,
    GLNORMErrorCode.E406_ALIAS_COLLISION: 409,

    # 422 Unprocessable Entity - Semantic errors
    GLNORMErrorCode.E300_CONVERSION_NOT_SUPPORTED: 422,
    GLNORMErrorCode.E301_MISSING_REFERENCE_CONDITIONS: 422,
    GLNORMErrorCode.E305_GWP_VERSION_MISSING: 422,
    GLNORMErrorCode.E306_BASIS_MISSING: 422,
    GLNORMErrorCode.E307_CONVERSION_FACTOR_DEPRECATED: 422,
    GLNORMErrorCode.E402_ENTITY_DEPRECATED: 422,
    GLNORMErrorCode.E403_LOW_CONFIDENCE_MATCH: 422,
    GLNORMErrorCode.E407_LLM_CANDIDATE_ONLY: 422,
    GLNORMErrorCode.E500_VOCABULARY_VERSION_MISMATCH: 422,
    GLNORMErrorCode.E503_VOCABULARY_EXPIRED: 422,
    GLNORMErrorCode.E504_GOVERNANCE_REQUIRED: 422,

    # 429 Too Many Requests - Rate limiting
    GLNORMErrorCode.E900_LIMIT_EXCEEDED: 429,

    # 500 Internal Server Error - Server errors
    GLNORMErrorCode.E304_PRECISION_OVERFLOW: 500,
    GLNORMErrorCode.E501_VOCABULARY_LOAD_FAILED: 500,
    GLNORMErrorCode.E502_VOCABULARY_CORRUPTED: 500,
    GLNORMErrorCode.E600_AUDIT_WRITE_FAILED: 500,
    GLNORMErrorCode.E602_AUDIT_INTEGRITY_VIOLATION: 500,
    GLNORMErrorCode.E603_REPLAY_MISMATCH: 500,
    GLNORMErrorCode.E902_RESOURCE_EXHAUSTED: 500,
    GLNORMErrorCode.E904_INTERNAL_ERROR: 500,

    # 503 Service Unavailable
    GLNORMErrorCode.E601_AUDIT_STORE_UNAVAILABLE: 503,
    GLNORMErrorCode.E903_SERVICE_UNAVAILABLE: 503,

    # 504 Gateway Timeout
    GLNORMErrorCode.E901_TIMEOUT: 504,
}


def get_error_code_by_value(value: str) -> Optional[GLNORMErrorCode]:
    """
    Look up an error code by its string value.

    Args:
        value: The error code string (e.g., "GLNORM-E100")

    Returns:
        GLNORMErrorCode if found, None otherwise

    Example:
        >>> code = get_error_code_by_value("GLNORM-E100")
        >>> assert code == GLNORMErrorCode.E100_UNIT_PARSE_FAILED
    """
    for code in GLNORMErrorCode:
        if code.value == value:
            return code
    return None


def get_codes_by_category(category: int) -> list:
    """
    Get all error codes in a specific category.

    Args:
        category: Category number (1-9)

    Returns:
        List of GLNORMErrorCode in that category

    Example:
        >>> codes = get_codes_by_category(1)
        >>> assert GLNORMErrorCode.E100_UNIT_PARSE_FAILED in codes
    """
    return [code for code in GLNORMErrorCode if code.category == category]


def get_all_codes() -> list:
    """
    Get all defined error codes.

    Returns:
        List of all GLNORMErrorCode values

    Example:
        >>> codes = get_all_codes()
        >>> assert len(codes) > 0
    """
    return list(GLNORMErrorCode)
