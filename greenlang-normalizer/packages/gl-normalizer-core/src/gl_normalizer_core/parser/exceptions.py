"""
Parser exceptions for GL-FOUND-X-003 Unit & Reference Normalizer.

This module defines specialized exceptions for the unit parsing subsystem,
providing structured error information with suggestions for remediation.

Error codes follow the GLNORM-Exxx taxonomy defined in gl_normalizer_core.errors.codes.

Example:
    >>> from gl_normalizer_core.parser.exceptions import UnitParseError
    >>> raise UnitParseError(
    ...     raw_unit="kiloWatt hours",
    ...     message="Failed to parse unit string",
    ...     suggestions=["kWh", "kW*h", "kilowatt_hour"]
    ... )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from gl_normalizer_core.errors.codes import GLNORMErrorCode


@dataclass
class ParserException(Exception):
    """
    Base exception for all parser-related errors.

    Provides structured error information including error code,
    message, and optional context for debugging.

    Attributes:
        code: GLNORM error code from the taxonomy
        message: Human-readable error message
        context: Additional context for debugging (optional)
        hint: Suggested remediation steps (optional)
        docs_url: Link to relevant documentation (optional)
    """

    code: GLNORMErrorCode
    message: str
    context: Optional[Dict[str, Any]] = None
    hint: Optional[str] = None
    docs_url: Optional[str] = None

    def __post_init__(self) -> None:
        """Initialize the exception with the message."""
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to a dictionary for JSON serialization.

        Returns:
            Dictionary representation of the exception

        Example:
            >>> exc = ParserException(
            ...     code=GLNORMErrorCode.E100_UNIT_PARSE_FAILED,
            ...     message="Parse error"
            ... )
            >>> d = exc.to_dict()
            >>> assert d["code"] == "GLNORM-E100"
        """
        result: Dict[str, Any] = {
            "code": self.code.value,
            "severity": self.code.severity,
            "message": self.message,
        }
        if self.context:
            result["context"] = self.context
        if self.hint:
            result["hint"] = self.hint
        if self.docs_url:
            result["docs"] = self.docs_url
        return result

    @property
    def http_status(self) -> int:
        """Return the appropriate HTTP status code for this exception."""
        return self.code.http_status


@dataclass
class UnitParseError(ParserException):
    """
    Exception raised when a unit string cannot be parsed.

    This exception is raised when the parser encounters a unit string
    that does not conform to any recognized pattern or contains
    invalid characters/structure.

    Attributes:
        raw_unit: The original unit string that failed to parse
        position: Character position where parsing failed (optional)
        suggestions: List of similar valid units the user might have meant

    Example:
        >>> raise UnitParseError(
        ...     raw_unit="kilo Watt per hour",
        ...     message="Unrecognized unit structure",
        ...     suggestions=["kW/h", "kWh", "kW*h"]
        ... )
    """

    raw_unit: str = ""
    position: Optional[int] = None
    suggestions: List[str] = field(default_factory=list)
    code: GLNORMErrorCode = field(
        default=GLNORMErrorCode.E100_UNIT_PARSE_FAILED
    )
    message: str = "Failed to parse unit string"

    def __post_init__(self) -> None:
        """Build the full error message with suggestions."""
        parts = [self.message]
        if self.raw_unit:
            parts.append(f"Input: '{self.raw_unit}'")
        if self.position is not None:
            parts.append(f"Position: {self.position}")
        if self.suggestions:
            parts.append(f"Did you mean: {', '.join(self.suggestions[:5])}")

        self.message = " | ".join(parts)
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with unit-specific fields."""
        result = super().to_dict()
        result["raw_unit"] = self.raw_unit
        if self.position is not None:
            result["position"] = self.position
        if self.suggestions:
            result["suggestions"] = self.suggestions[:10]
        return result


@dataclass
class AmbiguousUnitError(ParserException):
    """
    Exception raised when a unit string matches multiple interpretations.

    This exception is raised when the parser cannot determine a unique
    interpretation of a unit string, requiring user disambiguation.

    Attributes:
        raw_unit: The original ambiguous unit string
        candidates: List of possible unit interpretations with metadata
        locale_hint: If a locale was provided, which interpretation it suggests

    Example:
        >>> raise AmbiguousUnitError(
        ...     raw_unit="gallon",
        ...     candidates=[
        ...         {"unit": "gal_us", "name": "US gallon", "volume_L": 3.785},
        ...         {"unit": "gal_uk", "name": "Imperial gallon", "volume_L": 4.546}
        ...     ]
        ... )
    """

    raw_unit: str = ""
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    locale_hint: Optional[str] = None
    code: GLNORMErrorCode = field(default=GLNORMErrorCode.E104_AMBIGUOUS_UNIT)
    message: str = "Ambiguous unit requires disambiguation"

    def __post_init__(self) -> None:
        """Build the full error message with candidates."""
        parts = [self.message]
        if self.raw_unit:
            parts.append(f"Input: '{self.raw_unit}'")
        if self.candidates:
            candidate_names = [c.get("unit", str(c)) for c in self.candidates[:5]]
            parts.append(f"Possible interpretations: {', '.join(candidate_names)}")
        if self.locale_hint:
            parts.append(f"Locale suggestion: {self.locale_hint}")

        self.message = " | ".join(parts)
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with ambiguity-specific fields."""
        result = super().to_dict()
        result["raw_unit"] = self.raw_unit
        result["candidates"] = self.candidates
        if self.locale_hint:
            result["locale_hint"] = self.locale_hint
        return result


@dataclass
class LocaleRequiredError(ParserException):
    """
    Exception raised when locale information is required but not provided.

    This exception is raised when parsing or interpreting a unit or
    numeric value requires locale context that was not supplied.

    Attributes:
        raw_value: The value that requires locale context
        reason: Specific reason why locale is required
        supported_locales: List of supported locale identifiers

    Example:
        >>> raise LocaleRequiredError(
        ...     raw_value="1,234.56",
        ...     reason="Numeric format is ambiguous without locale",
        ...     supported_locales=["en_US", "de_DE", "fr_FR"]
        ... )
    """

    raw_value: str = ""
    reason: str = ""
    supported_locales: List[str] = field(default_factory=list)
    code: GLNORMErrorCode = field(default=GLNORMErrorCode.E106_LOCALE_PARSE_ERROR)
    message: str = "Locale information required"

    def __post_init__(self) -> None:
        """Build the full error message with locale context."""
        parts = [self.message]
        if self.raw_value:
            parts.append(f"Value: '{self.raw_value}'")
        if self.reason:
            parts.append(f"Reason: {self.reason}")
        if self.supported_locales:
            parts.append(f"Supported locales: {', '.join(self.supported_locales[:10])}")

        self.message = " | ".join(parts)
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with locale-specific fields."""
        result = super().to_dict()
        result["raw_value"] = self.raw_value
        result["reason"] = self.reason
        if self.supported_locales:
            result["supported_locales"] = self.supported_locales
        return result


@dataclass
class UnknownUnitError(ParserException):
    """
    Exception raised when a unit is not found in the registry.

    Attributes:
        raw_unit: The unrecognized unit string
        suggestions: List of similar valid units

    Example:
        >>> raise UnknownUnitError(
        ...     raw_unit="killowatt",
        ...     suggestions=["kilowatt", "kW", "kilowatt_hour"]
        ... )
    """

    raw_unit: str = ""
    suggestions: List[str] = field(default_factory=list)
    code: GLNORMErrorCode = field(default=GLNORMErrorCode.E101_UNKNOWN_UNIT)
    message: str = "Unit not found in registry"

    def __post_init__(self) -> None:
        """Build the full error message."""
        parts = [self.message]
        if self.raw_unit:
            parts.append(f"Input: '{self.raw_unit}'")
        if self.suggestions:
            parts.append(f"Did you mean: {', '.join(self.suggestions[:5])}")

        self.message = " | ".join(parts)
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with unit-specific fields."""
        result = super().to_dict()
        result["raw_unit"] = self.raw_unit
        if self.suggestions:
            result["suggestions"] = self.suggestions[:10]
        return result


@dataclass
class InvalidPrefixError(ParserException):
    """
    Exception raised when an unrecognized SI prefix is encountered.

    Attributes:
        prefix: The unrecognized prefix
        unit: The base unit the prefix was applied to
        valid_prefixes: List of valid prefixes for this unit

    Example:
        >>> raise InvalidPrefixError(
        ...     prefix="mega",
        ...     unit="gram",
        ...     valid_prefixes=["k", "M", "G", "m", "u", "n"]
        ... )
    """

    prefix: str = ""
    unit: str = ""
    valid_prefixes: List[str] = field(default_factory=list)
    code: GLNORMErrorCode = field(default=GLNORMErrorCode.E102_INVALID_PREFIX)
    message: str = "Invalid SI prefix"

    def __post_init__(self) -> None:
        """Build the full error message."""
        parts = [self.message]
        if self.prefix:
            parts.append(f"Prefix: '{self.prefix}'")
        if self.unit:
            parts.append(f"Unit: '{self.unit}'")
        if self.valid_prefixes:
            parts.append(f"Valid prefixes: {', '.join(self.valid_prefixes)}")

        self.message = " | ".join(parts)
        super().__post_init__()


@dataclass
class InvalidExponentError(ParserException):
    """
    Exception raised when an invalid exponent is encountered.

    Attributes:
        exponent: The invalid exponent value/string
        unit: The unit the exponent was applied to

    Example:
        >>> raise InvalidExponentError(
        ...     exponent="abc",
        ...     unit="m"
        ... )
    """

    exponent: str = ""
    unit: str = ""
    code: GLNORMErrorCode = field(default=GLNORMErrorCode.E103_INVALID_EXPONENT)
    message: str = "Invalid exponent notation"

    def __post_init__(self) -> None:
        """Build the full error message."""
        parts = [self.message]
        if self.exponent:
            parts.append(f"Exponent: '{self.exponent}'")
        if self.unit:
            parts.append(f"Unit: '{self.unit}'")

        self.message = " | ".join(parts)
        super().__post_init__()


@dataclass
class UnsupportedCompoundError(ParserException):
    """
    Exception raised when a compound unit structure is not supported.

    Attributes:
        raw_unit: The unsupported compound unit
        reason: Specific reason why this structure is not supported

    Example:
        >>> raise UnsupportedCompoundError(
        ...     raw_unit="kg/m/s/K",
        ...     reason="Nested divisions not supported"
        ... )
    """

    raw_unit: str = ""
    reason: str = ""
    code: GLNORMErrorCode = field(default=GLNORMErrorCode.E105_UNSUPPORTED_COMPOUND)
    message: str = "Unsupported compound unit structure"

    def __post_init__(self) -> None:
        """Build the full error message."""
        parts = [self.message]
        if self.raw_unit:
            parts.append(f"Input: '{self.raw_unit}'")
        if self.reason:
            parts.append(f"Reason: {self.reason}")

        self.message = " | ".join(parts)
        super().__post_init__()
