"""
Pint integration layer for GL-FOUND-X-003 Unit & Reference Normalizer.

This module provides a wrapper around the Pint library for unit parsing
and conversion, with GreenLang-specific extensions including:
- Custom units: Nm3, scf, CO2e, kgCO2e, tCO2e
- SI prefix configuration (k, M, G, m, u, n)
- Error handling with suggestions
- Provenance tracking for audit trails

Example:
    >>> from gl_normalizer_core.parser.pint_wrapper import PintUnitRegistry
    >>> registry = PintUnitRegistry()
    >>> result = registry.parse_unit("kgCO2e/kWh")
    >>> print(result.unit)
    >>> print(result.dimension_signature)
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import pint
    from pint import UnitRegistry, UndefinedUnitError, DimensionalityError
    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False
    UnitRegistry = None  # type: ignore
    UndefinedUnitError = Exception  # type: ignore
    DimensionalityError = Exception  # type: ignore

from gl_normalizer_core.parser.exceptions import (
    AmbiguousUnitError,
    InvalidPrefixError,
    UnitParseError,
    UnknownUnitError,
)
from gl_normalizer_core.parser.preprocessor import UnitPreprocessor, PreprocessResult
from gl_normalizer_core.parser.ast import UnitAST, UnitASTParser, UnitTerm

logger = logging.getLogger(__name__)


# =============================================================================
# GreenLang Custom Unit Definitions
# =============================================================================

# Custom unit definitions in Pint format
GREENLANG_UNIT_DEFINITIONS: List[str] = [
    # Standard volume conditions
    "Nm3 = m ** 3 = normal_cubic_meter",  # Normal cubic meter (0C, 101.325 kPa)
    "scf = 0.0283168 * m ** 3 = standard_cubic_foot",  # Standard cubic foot (60F, 14.73 psi)

    # Emissions units (CO2 equivalent - treated as mass)
    "CO2e = kg = carbon_dioxide_equivalent",
    "kgCO2e = kg = kilogram_CO2_equivalent",
    "tCO2e = 1000 * kg = tonne_CO2_equivalent = metric_ton_CO2e",
    "lbCO2e = 0.453592 * kg = pound_CO2_equivalent",
    "MtCO2e = 1e9 * kg = megatonne_CO2_equivalent",
    "GtCO2e = 1e12 * kg = gigatonne_CO2_equivalent",

    # Additional mass units
    "short_ton = 907.185 * kg = US_ton = ton_us",
    "long_ton = 1016.05 * kg = imperial_ton = ton_uk",
    "metric_tonne = 1000 * kg = tonne",

    # Energy units
    "MMBTU = 1055.06 * MJ = million_BTU",
    "thm = 105.506 * MJ = therm",

    # Volume variants
    "gal_us = 3.78541 * L = US_gallon = gallon_us",
    "gal_uk = 4.54609 * L = imperial_gallon = gallon_uk",
    "bbl = 158.987 * L = barrel = oil_barrel",
    "bbl_oil = 158.987 * L = petroleum_barrel",

    # Temperature aliases
    "degC = kelvin; offset: 273.15 = degree_Celsius = celsius",
    "degF = 5 / 9 * kelvin; offset: 255.372 = degree_Fahrenheit = fahrenheit",

    # GHG species (for reference, treated as dimensionless multipliers)
    "CO2 = 1 * CO2e = carbon_dioxide",
    "CH4 = 1 * CO2e = methane",  # Actual GWP applied separately
    "N2O = 1 * CO2e = nitrous_oxide",  # Actual GWP applied separately
    "SF6 = 1 * CO2e = sulfur_hexafluoride",  # Actual GWP applied separately
]

# Known unit symbols for suggestion matching
KNOWN_UNIT_SYMBOLS: Set[str] = {
    # SI base
    "m", "kg", "s", "A", "K", "mol", "cd",
    # SI derived
    "Hz", "N", "Pa", "J", "W", "C", "V", "F", "ohm", "S", "Wb", "T", "H", "lm", "lx", "Bq", "Gy", "Sv", "kat",
    # Common
    "g", "L", "l", "h", "min", "d", "a",
    # Energy
    "Wh", "kWh", "MWh", "GWh", "TWh",
    "kJ", "MJ", "GJ", "TJ",
    "BTU", "MMBTU", "thm",
    # Mass
    "t", "tonne", "lb", "oz",
    "short_ton", "long_ton", "metric_tonne",
    # Volume
    "mL", "kL", "ML",
    "gal", "gal_us", "gal_uk", "bbl", "scf", "Nm3",
    # Emissions
    "CO2e", "kgCO2e", "tCO2e", "lbCO2e", "MtCO2e", "GtCO2e",
    "CO2", "CH4", "N2O", "SF6",
    # Temperature
    "degC", "degF",
    # Pressure
    "bar", "mbar", "psi", "atm",
    # Area
    "ha", "acre",
    # Length
    "km", "cm", "mm", "um", "nm",
    "ft", "in", "yd", "mi",
}


@dataclass
class UnitParseResult:
    """
    Result of parsing a unit string.

    Attributes:
        success: Whether parsing was successful
        raw_unit: Original unit string
        normalized_unit: Normalized unit string
        pint_unit: Pint Unit object (if available)
        ast: UnitAST representation
        dimension_signature: Dictionary of dimension exponents
        conversion_factor_to_base: Factor to convert to base SI units
        parse_time_ms: Time taken to parse in milliseconds
        provenance_hash: SHA-256 hash for audit trail
        warnings: List of warnings generated during parsing
        error: Error message if parsing failed
        suggestions: Suggested corrections if parsing failed
    """

    success: bool
    raw_unit: str
    normalized_unit: str = ""
    pint_unit: Optional[Any] = None  # pint.Unit
    ast: Optional[UnitAST] = None
    dimension_signature: Dict[str, int] = field(default_factory=dict)
    conversion_factor_to_base: Optional[Decimal] = None
    parse_time_ms: float = 0.0
    provenance_hash: str = ""
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash and self.success:
            content = f"{self.raw_unit}|{self.normalized_unit}|{self.dimension_signature}"
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "success": self.success,
            "raw_unit": self.raw_unit,
            "normalized_unit": self.normalized_unit,
            "dimension_signature": self.dimension_signature,
            "parse_time_ms": self.parse_time_ms,
            "provenance_hash": self.provenance_hash,
        }
        if self.conversion_factor_to_base is not None:
            result["conversion_factor_to_base"] = str(self.conversion_factor_to_base)
        if self.ast:
            result["ast"] = self.ast.to_dict()
        if self.warnings:
            result["warnings"] = self.warnings
        if self.error:
            result["error"] = self.error
        if self.suggestions:
            result["suggestions"] = self.suggestions
        return result


class PintUnitRegistry:
    """
    GreenLang wrapper around Pint UnitRegistry.

    Provides unit parsing and conversion with GreenLang-specific extensions,
    error handling with suggestions, and audit trail support.

    Attributes:
        ureg: Pint UnitRegistry instance
        preprocessor: Unit string preprocessor
        ast_parser: AST parser for unit strings

    Example:
        >>> registry = PintUnitRegistry()
        >>> result = registry.parse_unit("kg*m/s**2")
        >>> print(result.dimension_signature)
        >>> # {"[mass]": 1, "[length]": 1, "[time]": -2}
    """

    def __init__(
        self,
        custom_definitions: Optional[List[str]] = None,
        enable_preprocessing: bool = True,
        cache_size: int = 10000,
    ) -> None:
        """
        Initialize the Pint wrapper with GreenLang extensions.

        Args:
            custom_definitions: Additional unit definitions to load
            enable_preprocessing: Whether to preprocess unit strings
            cache_size: Size of the unit parsing cache

        Raises:
            ImportError: If Pint is not available
        """
        if not PINT_AVAILABLE:
            raise ImportError(
                "Pint library is required. Install with: pip install pint"
            )

        # Initialize Pint registry
        self.ureg = pint.UnitRegistry(cache_folder=None)

        # Load GreenLang custom units
        self._load_greenlang_units()

        # Load additional custom definitions
        if custom_definitions:
            for definition in custom_definitions:
                try:
                    self.ureg.define(definition)
                except Exception as e:
                    logger.warning(f"Failed to load custom definition: {definition} - {e}")

        # Initialize preprocessor and AST parser
        self.preprocessor = UnitPreprocessor() if enable_preprocessing else None
        self.ast_parser = UnitASTParser()

        # Caching
        self._cache: Dict[str, UnitParseResult] = {}
        self._cache_size = cache_size

        # Known units for suggestions
        self._known_units = KNOWN_UNIT_SYMBOLS.copy()
        self._update_known_units()

    def _load_greenlang_units(self) -> None:
        """Load GreenLang custom unit definitions into Pint."""
        for definition in GREENLANG_UNIT_DEFINITIONS:
            try:
                self.ureg.define(definition)
                logger.debug(f"Loaded unit definition: {definition}")
            except Exception as e:
                logger.warning(f"Failed to load GreenLang unit: {definition} - {e}")

    def _update_known_units(self) -> None:
        """Update known units from the Pint registry."""
        try:
            # Add all units from Pint registry
            for unit_name in dir(self.ureg):
                if not unit_name.startswith("_"):
                    self._known_units.add(unit_name)
        except Exception as e:
            logger.warning(f"Failed to update known units from Pint: {e}")

    def parse_unit(
        self,
        unit_string: str,
        expected_dimension: Optional[str] = None,
    ) -> UnitParseResult:
        """
        Parse a unit string and return structured result.

        Args:
            unit_string: Unit string to parse
            expected_dimension: Expected dimension for validation (optional)

        Returns:
            UnitParseResult with parsing details

        Example:
            >>> registry = PintUnitRegistry()
            >>> result = registry.parse_unit("kgCO2e/kWh")
            >>> assert result.success
            >>> print(result.normalized_unit)
        """
        start_time = datetime.now()

        # Check cache
        cache_key = f"{unit_string}|{expected_dimension or ''}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            logger.debug(f"Cache hit for unit: {unit_string}")
            return cached

        try:
            # Preprocess if enabled
            preprocessed = unit_string
            warnings: List[str] = []

            if self.preprocessor:
                preprocess_result = self.preprocessor.preprocess(unit_string)
                preprocessed = preprocess_result.normalized
                if preprocess_result.warnings:
                    warnings.extend(preprocess_result.warnings)

            # Parse with Pint
            try:
                pint_unit = self.ureg.parse_expression(preprocessed)
                # Handle quantities vs units
                if hasattr(pint_unit, 'units'):
                    pint_unit = pint_unit.units
            except UndefinedUnitError as e:
                # Get suggestions for unknown unit
                suggestions = self._get_suggestions(unit_string)
                raise UnknownUnitError(
                    raw_unit=unit_string,
                    suggestions=suggestions
                ) from e
            except Exception as e:
                suggestions = self._get_suggestions(unit_string)
                raise UnitParseError(
                    raw_unit=unit_string,
                    message=str(e),
                    suggestions=suggestions
                ) from e

            # Build AST
            try:
                ast = self.ast_parser.parse(preprocessed)
            except Exception as e:
                logger.warning(f"AST parsing failed for {preprocessed}: {e}")
                ast = None

            # Get dimension signature from Pint
            dimension_signature = self._extract_dimensions(pint_unit)

            # Get conversion factor to base units
            conversion_factor = self._get_base_conversion_factor(pint_unit)

            # Validate dimension if expected
            if expected_dimension:
                self._validate_dimension(dimension_signature, expected_dimension)

            # Compute timing
            parse_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            result = UnitParseResult(
                success=True,
                raw_unit=unit_string,
                normalized_unit=str(pint_unit),
                pint_unit=pint_unit,
                ast=ast,
                dimension_signature=dimension_signature,
                conversion_factor_to_base=conversion_factor,
                parse_time_ms=parse_time_ms,
                warnings=warnings,
            )

            # Cache result
            self._cache_result(cache_key, result)

            return result

        except (UnitParseError, UnknownUnitError, AmbiguousUnitError) as e:
            parse_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            return UnitParseResult(
                success=False,
                raw_unit=unit_string,
                parse_time_ms=parse_time_ms,
                error=str(e),
                suggestions=getattr(e, 'suggestions', []),
            )
        except Exception as e:
            logger.error(f"Unexpected error parsing unit '{unit_string}': {e}", exc_info=True)
            parse_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            return UnitParseResult(
                success=False,
                raw_unit=unit_string,
                parse_time_ms=parse_time_ms,
                error=f"Unexpected error: {str(e)}",
                suggestions=self._get_suggestions(unit_string),
            )

    def _extract_dimensions(self, pint_unit: Any) -> Dict[str, int]:
        """
        Extract dimension signature from Pint unit.

        Args:
            pint_unit: Pint Unit object

        Returns:
            Dictionary mapping dimension names to exponents
        """
        try:
            dimensionality = pint_unit.dimensionality
            # Convert Pint dimensionality to our format
            result: Dict[str, int] = {}
            for dim, exp in dimensionality.items():
                # Pint uses [length], [mass], etc.
                dim_name = str(dim).strip("[]")
                if exp != 0:
                    result[dim_name] = int(exp)
            return result
        except Exception as e:
            logger.warning(f"Failed to extract dimensions: {e}")
            return {}

    def _get_base_conversion_factor(self, pint_unit: Any) -> Optional[Decimal]:
        """
        Get the conversion factor to base SI units.

        Args:
            pint_unit: Pint Unit object

        Returns:
            Decimal conversion factor or None
        """
        try:
            # Create a quantity of 1 and convert to base
            quantity = 1 * pint_unit
            base_quantity = quantity.to_base_units()
            return Decimal(str(base_quantity.magnitude))
        except Exception as e:
            logger.warning(f"Failed to get conversion factor: {e}")
            return None

    def _validate_dimension(
        self,
        actual: Dict[str, int],
        expected: str
    ) -> None:
        """
        Validate that actual dimension matches expected.

        Args:
            actual: Actual dimension signature
            expected: Expected dimension name

        Raises:
            UnitParseError: If dimensions don't match
        """
        # Map common dimension names to signatures
        dimension_map: Dict[str, Dict[str, int]] = {
            "mass": {"mass": 1},
            "length": {"length": 1},
            "time": {"time": 1},
            "energy": {"mass": 1, "length": 2, "time": -2},
            "power": {"mass": 1, "length": 2, "time": -3},
            "volume": {"length": 3},
            "area": {"length": 2},
            "velocity": {"length": 1, "time": -1},
            "acceleration": {"length": 1, "time": -2},
            "force": {"mass": 1, "length": 1, "time": -2},
            "pressure": {"mass": 1, "length": -1, "time": -2},
            "density": {"mass": 1, "length": -3},
            "emissions_mass": {"mass": 1},  # CO2e treated as mass
            "emissions_intensity": {"time": -2},  # kg/kWh simplified
        }

        expected_sig = dimension_map.get(expected.lower(), {})
        if expected_sig and actual != expected_sig:
            raise UnitParseError(
                raw_unit="",
                message=f"Dimension mismatch: expected {expected} {expected_sig}, got {actual}"
            )

    def _get_suggestions(self, unit_string: str) -> List[str]:
        """
        Get suggested corrections for an invalid unit string.

        Args:
            unit_string: The invalid unit string

        Returns:
            List of suggested valid unit strings
        """
        # Try to find close matches
        suggestions = get_close_matches(
            unit_string.lower(),
            [u.lower() for u in self._known_units],
            n=5,
            cutoff=0.6
        )

        # Map back to original case
        case_map = {u.lower(): u for u in self._known_units}
        return [case_map.get(s, s) for s in suggestions]

    def _cache_result(self, key: str, result: UnitParseResult) -> None:
        """Cache a parse result with LRU eviction."""
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry (simple FIFO, could use LRU)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = result

    def convert(
        self,
        value: Union[int, float, Decimal],
        from_unit: str,
        to_unit: str,
    ) -> Tuple[Decimal, Dict[str, Any]]:
        """
        Convert a value from one unit to another.

        Args:
            value: Numeric value to convert
            from_unit: Source unit string
            to_unit: Target unit string

        Returns:
            Tuple of (converted value, conversion trace dict)

        Raises:
            UnitParseError: If units cannot be parsed
            DimensionalityError: If units are incompatible

        Example:
            >>> registry = PintUnitRegistry()
            >>> value, trace = registry.convert(100, "kWh", "MJ")
            >>> print(value)  # Decimal('360')
        """
        # Parse both units
        from_result = self.parse_unit(from_unit)
        if not from_result.success:
            raise UnitParseError(
                raw_unit=from_unit,
                message=from_result.error or "Failed to parse source unit",
                suggestions=from_result.suggestions
            )

        to_result = self.parse_unit(to_unit)
        if not to_result.success:
            raise UnitParseError(
                raw_unit=to_unit,
                message=to_result.error or "Failed to parse target unit",
                suggestions=to_result.suggestions
            )

        # Perform conversion
        try:
            quantity = float(value) * from_result.pint_unit
            converted = quantity.to(to_result.pint_unit)
            converted_value = Decimal(str(converted.magnitude))

            # Build conversion trace
            trace: Dict[str, Any] = {
                "from_unit": from_unit,
                "to_unit": to_unit,
                "from_normalized": from_result.normalized_unit,
                "to_normalized": to_result.normalized_unit,
                "original_value": str(value),
                "converted_value": str(converted_value),
                "conversion_factor": str(converted_value / Decimal(str(value))) if value != 0 else "N/A",
            }

            return converted_value, trace

        except DimensionalityError as e:
            raise UnitParseError(
                raw_unit=f"{from_unit} -> {to_unit}",
                message=f"Incompatible dimensions: {e}"
            ) from e

    def is_compatible(self, unit1: str, unit2: str) -> bool:
        """
        Check if two units are dimensionally compatible.

        Args:
            unit1: First unit string
            unit2: Second unit string

        Returns:
            True if units can be converted to each other
        """
        result1 = self.parse_unit(unit1)
        result2 = self.parse_unit(unit2)

        if not result1.success or not result2.success:
            return False

        return result1.dimension_signature == result2.dimension_signature

    def get_dimension(self, unit_string: str) -> Optional[str]:
        """
        Get the dimension name for a unit.

        Args:
            unit_string: Unit string

        Returns:
            Dimension name or None if not recognized
        """
        result = self.parse_unit(unit_string)
        if not result.success:
            return None

        # Map signature to dimension name
        sig = result.dimension_signature
        dimension_names: Dict[str, str] = {
            "mass:1": "mass",
            "length:1": "length",
            "time:1": "time",
            "mass:1,length:2,time:-2": "energy",
            "mass:1,length:2,time:-3": "power",
            "length:3": "volume",
            "length:2": "area",
        }

        # Create signature key
        sig_key = ",".join(f"{k}:{v}" for k, v in sorted(sig.items()))
        return dimension_names.get(sig_key)

    def clear_cache(self) -> None:
        """Clear the unit parsing cache."""
        self._cache.clear()
        logger.info("Unit parsing cache cleared")


# Module-level convenience instance
_default_registry: Optional[PintUnitRegistry] = None


def get_registry() -> PintUnitRegistry:
    """
    Get the default PintUnitRegistry instance.

    Returns:
        Singleton PintUnitRegistry instance
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = PintUnitRegistry()
    return _default_registry


def parse_unit(unit_string: str) -> UnitParseResult:
    """
    Convenience function to parse a unit string.

    Args:
        unit_string: Unit string to parse

    Returns:
        UnitParseResult

    Example:
        >>> result = parse_unit("kg/m**3")
        >>> print(result.dimension_signature)
    """
    return get_registry().parse_unit(unit_string)


def convert_unit(
    value: Union[int, float, Decimal],
    from_unit: str,
    to_unit: str,
) -> Decimal:
    """
    Convenience function to convert a value between units.

    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted value as Decimal

    Example:
        >>> result = convert_unit(100, "kWh", "MJ")
        >>> print(result)  # Decimal('360')
    """
    converted, _ = get_registry().convert(value, from_unit, to_unit)
    return converted
