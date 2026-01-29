"""
Unit Parser module for GL-FOUND-X-003 Unit & Reference Normalizer.

This module provides comprehensive unit parsing capabilities for the
GreenLang Normalizer, including:

- **Pint Integration** (pint_wrapper.py): Unit parsing and conversion using
  the Pint library with GreenLang-specific extensions for emissions units.

- **Preprocessing** (preprocessor.py): Unicode normalization, whitespace
  handling, synonym expansion, and exponent notation normalization.

- **AST Generation** (ast.py): Abstract syntax tree representation of unit
  expressions with dimension signature computation.

- **Locale Handling** (locale.py): Locale-aware resolution of ambiguous
  units and numeric string parsing.

- **Exceptions** (exceptions.py): Structured exceptions with error codes,
  suggestions, and audit-ready serialization.

Typical Usage:
    >>> from gl_normalizer_core.parser import parse_unit, convert_unit
    >>> result = parse_unit("kgCO2e/kWh")
    >>> assert result.success
    >>> print(result.dimension_signature)

    >>> value = convert_unit(100, "kWh", "MJ")
    >>> print(value)  # Decimal('360')

For locale-aware parsing:
    >>> from gl_normalizer_core.parser import LocaleHandler, LocaleProfile
    >>> handler = LocaleHandler()
    >>> profile = LocaleProfile(locale_code="en_GB")
    >>> result = handler.resolve_ambiguous_unit("gallon", profile)
    >>> print(result.resolved)  # "gal_uk"

For preprocessing only:
    >>> from gl_normalizer_core.parser import preprocess_unit
    >>> result = preprocess_unit("kilograms per cubic metre")
    >>> print(result.normalized)  # "kg/m**3"

For AST generation:
    >>> from gl_normalizer_core.parser import parse_unit_to_ast
    >>> ast = parse_unit_to_ast("kg*m/s**2")
    >>> print(ast.dimension_signature)  # {"mass": 1, "length": 1, "time": -2}

Legacy API (preserved for backwards compatibility):
    >>> from gl_normalizer_core.parser import UnitParser
    >>> parser = UnitParser()
    >>> result = parser.parse("100 kg CO2e")
    >>> print(result.quantity.magnitude)
    100.0
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from decimal import Decimal
from enum import Enum
import re
import hashlib
from datetime import datetime

from pydantic import BaseModel, Field, field_validator
import structlog

from gl_normalizer_core.errors import ParseError

logger = structlog.get_logger(__name__)


# =============================================================================
# Legacy API Classes (preserved for backwards compatibility)
# =============================================================================

class UnitSystem(str, Enum):
    """Supported unit systems."""

    SI = "SI"
    IMPERIAL = "imperial"
    CUSTOM = "custom"


class Quantity(BaseModel):
    """
    Represents a parsed quantity with magnitude and unit.

    Attributes:
        magnitude: Numeric value of the quantity
        unit: Unit string (normalized)
        original_unit: Original unit string as provided
        uncertainty: Optional uncertainty value
        unit_system: The unit system (SI, imperial, custom)
    """

    magnitude: float = Field(..., description="Numeric value")
    unit: str = Field(..., description="Normalized unit string")
    original_unit: Optional[str] = Field(None, description="Original unit as provided")
    uncertainty: Optional[float] = Field(None, ge=0, description="Uncertainty value")
    unit_system: UnitSystem = Field(default=UnitSystem.SI, description="Unit system")

    @field_validator("magnitude", mode="before")
    @classmethod
    def validate_magnitude(cls, v: Any) -> float:
        """Validate and convert magnitude to float."""
        if isinstance(v, (int, float, Decimal)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v.replace(",", ""))
            except ValueError:
                raise ValueError(f"Invalid magnitude value: {v}")
        raise ValueError(f"Cannot convert {type(v)} to magnitude")

    def to_tuple(self) -> Tuple[float, str]:
        """Return (magnitude, unit) tuple."""
        return (self.magnitude, self.unit)

    def __str__(self) -> str:
        """Return string representation."""
        return f"{self.magnitude} {self.unit}"

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"Quantity(magnitude={self.magnitude}, unit='{self.unit}')"


class ParseResult(BaseModel):
    """
    Result of parsing a quantity string (legacy API).

    Attributes:
        success: Whether parsing succeeded
        quantity: The parsed quantity (if successful)
        original_input: The original input string
        parse_time_ms: Time taken to parse in milliseconds
        warnings: Any warnings generated during parsing
        provenance_hash: SHA-256 hash for audit trail
    """

    success: bool = Field(..., description="Whether parsing succeeded")
    quantity: Optional[Quantity] = Field(None, description="Parsed quantity")
    original_input: str = Field(..., description="Original input string")
    parse_time_ms: float = Field(..., description="Parse time in milliseconds")
    warnings: List[str] = Field(default_factory=list, description="Parse warnings")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")

    @classmethod
    def create_success(
        cls,
        quantity: Quantity,
        original_input: str,
        parse_time_ms: float,
        warnings: Optional[List[str]] = None,
    ) -> "ParseResult":
        """Create a successful parse result."""
        provenance_str = f"{original_input}|{quantity.magnitude}|{quantity.unit}"
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()
        return cls(
            success=True,
            quantity=quantity,
            original_input=original_input,
            parse_time_ms=parse_time_ms,
            warnings=warnings or [],
            provenance_hash=provenance_hash,
        )

    @classmethod
    def create_failure(
        cls,
        original_input: str,
        parse_time_ms: float,
        warnings: Optional[List[str]] = None,
    ) -> "ParseResult":
        """Create a failed parse result."""
        provenance_hash = hashlib.sha256(f"FAILED|{original_input}".encode()).hexdigest()
        return cls(
            success=False,
            quantity=None,
            original_input=original_input,
            parse_time_ms=parse_time_ms,
            warnings=warnings or [],
            provenance_hash=provenance_hash,
        )


class UnitParser:
    """
    Legacy parser for quantity strings (preserved for backwards compatibility).

    For new code, prefer using the new parse_unit() function from pint_wrapper.

    Attributes:
        aliases: Dictionary of unit aliases
        strict_mode: Whether to raise errors on ambiguous parses

    Example:
        >>> parser = UnitParser()
        >>> result = parser.parse("100 kg CO2e")
        >>> print(result.quantity)
        Quantity(magnitude=100.0, unit='kg_CO2e')
    """

    # Common unit aliases for sustainability reporting
    DEFAULT_ALIASES: Dict[str, str] = {
        # Mass
        "kg": "kilogram",
        "g": "gram",
        "mg": "milligram",
        "t": "metric_ton",
        "tonne": "metric_ton",
        "ton": "metric_ton",
        "lb": "pound",
        "lbs": "pound",
        # Energy
        "kWh": "kilowatt_hour",
        "kwh": "kilowatt_hour",
        "MWh": "megawatt_hour",
        "mwh": "megawatt_hour",
        "GWh": "gigawatt_hour",
        "J": "joule",
        "kJ": "kilojoule",
        "MJ": "megajoule",
        "GJ": "gigajoule",
        "TJ": "terajoule",
        "BTU": "british_thermal_unit",
        "btu": "british_thermal_unit",
        "therm": "therm",
        # Volume
        "L": "liter",
        "l": "liter",
        "mL": "milliliter",
        "ml": "milliliter",
        "m3": "cubic_meter",
        "m^3": "cubic_meter",
        "gal": "gallon",
        "bbl": "barrel",
        # Emissions
        "CO2e": "CO2_equivalent",
        "CO2eq": "CO2_equivalent",
        "CO2-eq": "CO2_equivalent",
        "tCO2e": "metric_ton_CO2_equivalent",
        "tCO2eq": "metric_ton_CO2_equivalent",
        "kgCO2e": "kilogram_CO2_equivalent",
        "kgCO2eq": "kilogram_CO2_equivalent",
        # Distance
        "km": "kilometer",
        "m": "meter",
        "mi": "mile",
        "nmi": "nautical_mile",
    }

    # Regex patterns for parsing
    QUANTITY_PATTERN = re.compile(
        r"^\s*"
        r"(?P<magnitude>[+-]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?)"
        r"\s*"
        r"(?P<unit>.+?)"
        r"\s*$"
    )

    COMPOUND_UNIT_PATTERN = re.compile(
        r"(?P<prefix>[a-zA-Z]+)?\s*(?P<base>[a-zA-Z0-9]+)\s*(?P<suffix>CO2e?|eq)?"
    )

    def __init__(
        self,
        aliases: Optional[Dict[str, str]] = None,
        strict_mode: bool = False,
    ) -> None:
        """
        Initialize UnitParser.

        Args:
            aliases: Custom unit aliases to add to defaults
            strict_mode: Raise errors on ambiguous parses
        """
        self.aliases = {**self.DEFAULT_ALIASES}
        if aliases:
            self.aliases.update(aliases)
        self.strict_mode = strict_mode
        logger.info("UnitParser initialized", strict_mode=strict_mode, alias_count=len(self.aliases))

    def parse(self, input_string: str) -> ParseResult:
        """
        Parse a quantity string.

        Args:
            input_string: String to parse (e.g., "100 kg CO2e")

        Returns:
            ParseResult containing the parsed quantity or error information

        Raises:
            ParseError: If strict_mode is True and parsing fails
        """
        start_time = datetime.now()
        warnings: List[str] = []

        try:
            # Validate input
            if not input_string or not input_string.strip():
                raise ParseError("Empty input string", input_value=input_string)

            # Match pattern
            match = self.QUANTITY_PATTERN.match(input_string.strip())
            if not match:
                raise ParseError(
                    "Could not parse quantity string",
                    input_value=input_string,
                    hint="Expected format: '<number> <unit>' (e.g., '100 kg')",
                )

            # Extract components
            magnitude_str = match.group("magnitude")
            unit_str = match.group("unit").strip()

            # Parse magnitude
            try:
                magnitude = float(magnitude_str.replace(",", ""))
            except ValueError as e:
                raise ParseError(
                    f"Invalid magnitude: {magnitude_str}",
                    input_value=input_string,
                    details={"magnitude_str": magnitude_str},
                ) from e

            # Normalize unit
            normalized_unit = self._normalize_unit(unit_str)

            # Detect unit system
            unit_system = self._detect_unit_system(normalized_unit)

            # Create quantity
            quantity = Quantity(
                magnitude=magnitude,
                unit=normalized_unit,
                original_unit=unit_str,
                unit_system=unit_system,
            )

            # Calculate parse time
            parse_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            logger.debug(
                "Parsed quantity",
                input=input_string,
                magnitude=magnitude,
                unit=normalized_unit,
                parse_time_ms=parse_time_ms,
            )

            return ParseResult.create_success(
                quantity=quantity,
                original_input=input_string,
                parse_time_ms=parse_time_ms,
                warnings=warnings,
            )

        except ParseError:
            if self.strict_mode:
                raise
            parse_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            return ParseResult.create_failure(
                original_input=input_string,
                parse_time_ms=parse_time_ms,
                warnings=warnings,
            )

    def _normalize_unit(self, unit_str: str) -> str:
        """
        Normalize a unit string.

        Args:
            unit_str: Raw unit string

        Returns:
            Normalized unit string
        """
        # Remove extra whitespace and normalize
        unit = unit_str.strip()

        # Handle compound units (e.g., "kg CO2e" -> "kg_CO2e")
        unit = re.sub(r"\s+", "_", unit)

        # Check aliases
        if unit in self.aliases:
            return self.aliases[unit]

        # Check case-insensitive aliases
        unit_lower = unit.lower()
        for alias, canonical in self.aliases.items():
            if alias.lower() == unit_lower:
                return canonical

        return unit

    def _detect_unit_system(self, unit: str) -> UnitSystem:
        """
        Detect the unit system for a normalized unit.

        Args:
            unit: Normalized unit string

        Returns:
            UnitSystem enum value
        """
        imperial_units = {"pound", "mile", "gallon", "british_thermal_unit", "barrel"}
        si_units = {
            "kilogram", "gram", "meter", "kilometer", "liter", "joule",
            "kilowatt_hour", "megawatt_hour", "metric_ton",
        }

        unit_lower = unit.lower()

        for si_unit in si_units:
            if si_unit in unit_lower:
                return UnitSystem.SI

        for imperial_unit in imperial_units:
            if imperial_unit in unit_lower:
                return UnitSystem.IMPERIAL

        return UnitSystem.CUSTOM

    def add_alias(self, alias: str, canonical: str) -> None:
        """
        Add a unit alias.

        Args:
            alias: Alias string
            canonical: Canonical unit name
        """
        self.aliases[alias] = canonical
        logger.debug("Added unit alias", alias=alias, canonical=canonical)

    def remove_alias(self, alias: str) -> bool:
        """
        Remove a unit alias.

        Args:
            alias: Alias to remove

        Returns:
            True if alias was removed, False if not found
        """
        if alias in self.aliases:
            del self.aliases[alias]
            logger.debug("Removed unit alias", alias=alias)
            return True
        return False


# =============================================================================
# New API Imports (GL-FOUND-X-003)
# =============================================================================

# Import from submodules
from gl_normalizer_core.parser.exceptions import (
    AmbiguousUnitError,
    InvalidExponentError,
    InvalidPrefixError,
    LocaleRequiredError,
    ParserException,
    UnitParseError,
    UnknownUnitError,
    UnsupportedCompoundError,
)

from gl_normalizer_core.parser.preprocessor import (
    PreprocessResult,
    UnitPreprocessor,
    preprocess_unit,
    UNIT_SYNONYMS,
    SEPARATOR_MAPPINGS,
    SUPERSCRIPT_MAP,
)

from gl_normalizer_core.parser.ast import (
    BASE_UNIT_DIMENSIONS,
    NO_PREFIX_UNITS,
    PREFIXABLE_UNITS,
    SI_PREFIX_NAMES,
    SI_PREFIXES,
    UnitAST,
    UnitASTParser,
    UnitTerm,
    compute_dimension_signature,
    parse_unit_to_ast,
)

from gl_normalizer_core.parser.locale import (
    AMBIGUOUS_UNITS,
    DECIMAL_SEPARATORS,
    SUPPORTED_LOCALES,
    LocaleHandler,
    LocaleProfile,
    LocaleResolutionLevel,
    LocaleResolutionResult,
    parse_number,
    resolve_unit_locale,
)

from gl_normalizer_core.parser.pint_wrapper import (
    GREENLANG_UNIT_DEFINITIONS,
    KNOWN_UNIT_SYMBOLS,
    PintUnitRegistry,
    UnitParseResult,
    convert_unit,
    get_registry,
    parse_unit,
)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Legacy API (backwards compatibility)
    "UnitParser",
    "Quantity",
    "ParseResult",
    "UnitSystem",
    # Exceptions
    "AmbiguousUnitError",
    "InvalidExponentError",
    "InvalidPrefixError",
    "LocaleRequiredError",
    "ParserException",
    "UnitParseError",
    "UnknownUnitError",
    "UnsupportedCompoundError",
    # Preprocessor
    "PreprocessResult",
    "UnitPreprocessor",
    "preprocess_unit",
    "UNIT_SYNONYMS",
    "SEPARATOR_MAPPINGS",
    "SUPERSCRIPT_MAP",
    # AST
    "BASE_UNIT_DIMENSIONS",
    "NO_PREFIX_UNITS",
    "PREFIXABLE_UNITS",
    "SI_PREFIX_NAMES",
    "SI_PREFIXES",
    "UnitAST",
    "UnitASTParser",
    "UnitTerm",
    "compute_dimension_signature",
    "parse_unit_to_ast",
    # Locale
    "AMBIGUOUS_UNITS",
    "DECIMAL_SEPARATORS",
    "SUPPORTED_LOCALES",
    "LocaleHandler",
    "LocaleProfile",
    "LocaleResolutionLevel",
    "LocaleResolutionResult",
    "parse_number",
    "resolve_unit_locale",
    # Pint Wrapper
    "GREENLANG_UNIT_DEFINITIONS",
    "KNOWN_UNIT_SYMBOLS",
    "PintUnitRegistry",
    "UnitParseResult",
    "convert_unit",
    "get_registry",
    "parse_unit",
]


# Module version
__version__ = "1.0.0"
