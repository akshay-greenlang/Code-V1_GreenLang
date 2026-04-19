"""
CBAM Pack Error Taxonomy

Defines all error types with codes following the PRD specification:
- VAL-001 to VAL-010: Validation errors
- CALC-001 to CALC-004: Calculation errors
- XML-001 to XML-003: Export errors
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ErrorCategory(str, Enum):
    """Error category classification."""
    SCHEMA = "Schema"
    BUSINESS = "Business Rule Violation"
    REFERENTIAL = "Referential Integrity"
    CONFIG = "Configuration"
    FACTOR = "Emission Factor"
    CONVERSION = "Unit Conversion"
    METHOD = "Method Selection"
    XML_SCHEMA = "XML Schema"
    ENCODING = "Encoding"
    STRUCTURE = "Structure"


@dataclass
class ErrorLocation:
    """Location of an error in source data."""
    file: str
    row: Optional[int] = None
    column: Optional[str] = None

    def __str__(self) -> str:
        parts = [self.file]
        if self.row is not None:
            parts.append(str(self.row))
        if self.column is not None:
            parts.append(self.column)
        return ":".join(parts)


class CBAMError(Exception):
    """Base exception for all CBAM Pack errors."""

    code: str = "CBAM-000"
    category: ErrorCategory = ErrorCategory.SCHEMA

    def __init__(
        self,
        message: str,
        location: Optional[ErrorLocation] = None,
        fix_guidance: Optional[str] = None,
    ):
        self.message = message
        self.location = location
        self.fix_guidance = fix_guidance
        super().__init__(self.format_error())

    def format_error(self) -> str:
        """Format error message following PRD specification."""
        lines = [f"Error: {self.code} {self.category.value}"]

        if self.location:
            lines.append(f"Location: {self.location}")

        lines.append(f"Problem: {self.message}")

        if self.fix_guidance:
            lines.append(f"Fix: {self.fix_guidance}")

        return "\n".join(lines)


# =============================================================================
# Validation Errors (VAL-001 to VAL-010)
# =============================================================================

class ValidationError(CBAMError):
    """Base class for validation errors."""
    pass


class MissingRequiredColumnError(ValidationError):
    """VAL-001: Missing required column in import file."""
    code = "VAL-001"
    category = ErrorCategory.SCHEMA

    def __init__(self, column_name: str, location: Optional[ErrorLocation] = None):
        super().__init__(
            message=f"Missing required column '{column_name}'",
            location=location,
            fix_guidance=f"Add the '{column_name}' column to your import file.",
        )


class InvalidDataTypeError(ValidationError):
    """VAL-002: Invalid data type for field."""
    code = "VAL-002"
    category = ErrorCategory.SCHEMA

    def __init__(
        self,
        field: str,
        expected_type: str,
        actual_value: str,
        location: Optional[ErrorLocation] = None,
    ):
        super().__init__(
            message=f"Expected {expected_type}, got '{actual_value}'",
            location=location,
            fix_guidance=f"Ensure '{field}' contains a valid {expected_type} value.",
        )


class InvalidEnumValueError(ValidationError):
    """VAL-003: Invalid enum value."""
    code = "VAL-003"
    category = ErrorCategory.SCHEMA

    def __init__(
        self,
        field: str,
        value: str,
        allowed_values: list[str],
        location: Optional[ErrorLocation] = None,
    ):
        allowed_str = ", ".join(allowed_values)
        super().__init__(
            message=f"Invalid value '{value}'. Expected: {allowed_str}",
            location=location,
            fix_guidance=f"Use one of the allowed values: {allowed_str}",
        )


class InvalidCNCodeFormatError(ValidationError):
    """VAL-004: Invalid CN code format."""
    code = "VAL-004"
    category = ErrorCategory.BUSINESS

    def __init__(self, cn_code: str, location: Optional[ErrorLocation] = None):
        super().__init__(
            message=f"CN code '{cn_code}' is not 8 digits. CBAM requires 8-digit CN codes.",
            location=location,
            fix_guidance="Verify the CN code in the EU Combined Nomenclature database.",
        )


class UnsupportedCNCodeError(ValidationError):
    """VAL-005: CN code not supported in MVP."""
    code = "VAL-005"
    category = ErrorCategory.BUSINESS

    def __init__(self, cn_code: str, category: str, location: Optional[ErrorLocation] = None):
        super().__init__(
            message=f"CN code '{cn_code}' is {category}, not supported in MVP. "
                    f"Only Steel (72xx, 73xx) and Aluminum (76xx) are supported.",
            location=location,
            fix_guidance="Use CN codes starting with 72, 73, or 76 for the MVP.",
        )


class InvalidCountryCodeError(ValidationError):
    """VAL-006: Invalid ISO country code."""
    code = "VAL-006"
    category = ErrorCategory.BUSINESS

    def __init__(self, country_code: str, location: Optional[ErrorLocation] = None):
        super().__init__(
            message=f"'{country_code}' is not a valid ISO 3166-1 alpha-2 country code.",
            location=location,
            fix_guidance="Use 2-letter ISO country codes (e.g., CN, DE, US, IN).",
        )


class NegativeQuantityError(ValidationError):
    """VAL-007: Negative quantity."""
    code = "VAL-007"
    category = ErrorCategory.BUSINESS

    def __init__(self, quantity: str, location: Optional[ErrorLocation] = None):
        super().__init__(
            message=f"Quantity '{quantity}' is not valid. Must be positive.",
            location=location,
            fix_guidance="Enter a positive number for quantity.",
        )


class UnknownUnitError(ValidationError):
    """VAL-008: Unknown or unsupported unit."""
    code = "VAL-008"
    category = ErrorCategory.BUSINESS

    def __init__(self, unit: str, location: Optional[ErrorLocation] = None):
        super().__init__(
            message=f"Unit '{unit}' is not supported. Use kg or tonnes.",
            location=location,
            fix_guidance="Convert your quantities to kg or tonnes.",
        )


class DuplicateLineIdError(ValidationError):
    """VAL-009: Duplicate line_id."""
    code = "VAL-009"
    category = ErrorCategory.REFERENTIAL

    def __init__(
        self,
        line_id: str,
        original_row: int,
        duplicate_row: int,
        location: Optional[ErrorLocation] = None,
    ):
        super().__init__(
            message=f"line_id '{line_id}' at row {duplicate_row} duplicates row {original_row}.",
            location=location,
            fix_guidance="Ensure each line_id is unique within the import file.",
        )


class MissingConfigFieldError(ValidationError):
    """VAL-010: Missing required config field."""
    code = "VAL-010"
    category = ErrorCategory.CONFIG

    def __init__(self, field_path: str, location: Optional[ErrorLocation] = None):
        super().__init__(
            message=f"'{field_path}' is required.",
            location=location,
            fix_guidance=f"Add the '{field_path}' field to your config file.",
        )


# =============================================================================
# Calculation Errors (CALC-001 to CALC-004)
# =============================================================================

class CalculationError(CBAMError):
    """Base class for calculation errors."""
    pass


class MissingEmissionFactorError(CalculationError):
    """CALC-001: No emission factor found."""
    code = "CALC-001"
    category = ErrorCategory.FACTOR

    def __init__(
        self,
        cn_code: str,
        country: str,
        location: Optional[ErrorLocation] = None,
    ):
        super().__init__(
            message=f"No emission factor for CN code '{cn_code}' from country '{country}'.",
            location=location,
            fix_guidance="Provide supplier-specific emissions data or verify the CN code/country.",
        )


class ExpiredEmissionFactorError(CalculationError):
    """CALC-002: Emission factor has expired."""
    code = "CALC-002"
    category = ErrorCategory.FACTOR

    def __init__(
        self,
        factor_id: str,
        expiry_date: str,
        location: Optional[ErrorLocation] = None,
    ):
        super().__init__(
            message=f"Emission factor '{factor_id}' expired on {expiry_date}.",
            location=location,
            fix_guidance="Update to the latest emission factor library version.",
        )


class UnitConversionError(CalculationError):
    """CALC-003: Cannot convert between units."""
    code = "CALC-003"
    category = ErrorCategory.CONVERSION

    def __init__(
        self,
        from_unit: str,
        to_unit: str,
        location: Optional[ErrorLocation] = None,
    ):
        super().__init__(
            message=f"Cannot convert '{from_unit}' to '{to_unit}'.",
            location=location,
            fix_guidance="Use supported units: kg, tonnes.",
        )


class InvalidMethodSelectionError(CalculationError):
    """CALC-004: Invalid calculation method selection."""
    code = "CALC-004"
    category = ErrorCategory.METHOD

    def __init__(self, reason: str, location: Optional[ErrorLocation] = None):
        super().__init__(
            message=f"Supplier data incomplete for 'supplier_specific' method: {reason}",
            location=location,
            fix_guidance="Provide complete supplier emissions data or use default factors.",
        )


# =============================================================================
# Export Errors (XML-001 to XML-003)
# =============================================================================

class ExportError(CBAMError):
    """Base class for export errors."""
    pass


class XSDValidationError(ExportError):
    """XML-001: XSD validation failed."""
    code = "XML-001"
    category = ErrorCategory.XML_SCHEMA

    def __init__(
        self,
        details: str,
        line: Optional[int] = None,
        location: Optional[ErrorLocation] = None,
    ):
        message = f"XSD validation failed: {details}"
        if line:
            message += f" at line {line}"
        super().__init__(
            message=message,
            location=location,
            fix_guidance="Check that all required XML elements are present and valid.",
        )


class EncodingError(ExportError):
    """XML-002: Character encoding error."""
    code = "XML-002"
    category = ErrorCategory.ENCODING

    def __init__(
        self,
        field: str,
        character: str,
        location: Optional[ErrorLocation] = None,
    ):
        super().__init__(
            message=f"Invalid character '{character}' in {field}.",
            location=location,
            fix_guidance="Use only UTF-8 compatible characters in your data.",
        )


class MissingRequiredElementError(ExportError):
    """XML-003: Missing required XML element."""
    code = "XML-003"
    category = ErrorCategory.STRUCTURE

    def __init__(self, element_path: str, location: Optional[ErrorLocation] = None):
        super().__init__(
            message=f"Required element '{element_path}' is missing.",
            location=location,
            fix_guidance=f"Ensure '{element_path}' is provided in your configuration.",
        )
