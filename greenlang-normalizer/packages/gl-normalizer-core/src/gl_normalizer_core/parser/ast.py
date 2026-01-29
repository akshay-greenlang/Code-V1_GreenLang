"""
Unit AST (Abstract Syntax Tree) generator for GL-FOUND-X-003.

This module provides data structures and parsing logic to represent
unit expressions as an abstract syntax tree. The AST captures:
- Numerator and denominator terms
- Exponents for each term
- SI prefixes
- Dimension signatures

Example:
    >>> from gl_normalizer_core.parser.ast import parse_unit_to_ast
    >>> ast = parse_unit_to_ast("kg*m/s**2")
    >>> print(ast.numerator_terms)  # [UnitTerm(base="kg", exponent=1), UnitTerm(base="m", exponent=1)]
    >>> print(ast.denominator_terms)  # [UnitTerm(base="s", exponent=2)]
    >>> print(ast.dimension_signature)  # {"mass": 1, "length": 1, "time": -2}
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

from gl_normalizer_core.parser.exceptions import (
    InvalidExponentError,
    InvalidPrefixError,
    UnitParseError,
    UnsupportedCompoundError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SI Prefix Definitions
# =============================================================================

SI_PREFIXES: Dict[str, Decimal] = {
    "Y": Decimal("1e24"),   # yotta
    "Z": Decimal("1e21"),   # zetta
    "E": Decimal("1e18"),   # exa
    "P": Decimal("1e15"),   # peta
    "T": Decimal("1e12"),   # tera
    "G": Decimal("1e9"),    # giga
    "M": Decimal("1e6"),    # mega
    "k": Decimal("1e3"),    # kilo
    "h": Decimal("1e2"),    # hecto
    "da": Decimal("1e1"),   # deca
    "d": Decimal("1e-1"),   # deci
    "c": Decimal("1e-2"),   # centi
    "m": Decimal("1e-3"),   # milli
    "u": Decimal("1e-6"),   # micro (ASCII substitute)
    "\u03bc": Decimal("1e-6"),  # micro (Greek mu)
    "\u00b5": Decimal("1e-6"),  # micro (micro sign)
    "n": Decimal("1e-9"),   # nano
    "p": Decimal("1e-12"),  # pico
    "f": Decimal("1e-15"),  # femto
    "a": Decimal("1e-18"),  # atto
    "z": Decimal("1e-21"),  # zepto
    "y": Decimal("1e-24"),  # yocto
}

SI_PREFIX_NAMES: Dict[str, str] = {
    "Y": "yotta",
    "Z": "zetta",
    "E": "exa",
    "P": "peta",
    "T": "tera",
    "G": "giga",
    "M": "mega",
    "k": "kilo",
    "h": "hecto",
    "da": "deca",
    "d": "deci",
    "c": "centi",
    "m": "milli",
    "u": "micro",
    "\u03bc": "micro",
    "\u00b5": "micro",
    "n": "nano",
    "p": "pico",
    "f": "femto",
    "a": "atto",
    "z": "zepto",
    "y": "yocto",
}

# Base units that accept SI prefixes
PREFIXABLE_UNITS: Set[str] = {
    "g", "m", "s", "A", "K", "mol", "cd",  # SI base
    "L", "l", "W", "J", "N", "Pa", "Hz", "V", "F", "ohm",  # SI derived
    "Wh", "J",  # Energy
    "t", "tonne",  # Metric tonne
    "bar",  # Pressure
}

# Units that should NOT have prefixes stripped
NO_PREFIX_UNITS: Set[str] = {
    "min", "mile", "mi", "mol", "month",  # Starts with 'm' but not milli
    "ha", "hour", "hr",  # Starts with 'h' but not hecto
    "day", "deg", "degC", "degF",  # Starts with 'd' but not deci
    "year", "yr",  # Starts with 'y' but not yocto
    "ft", "in", "lb", "oz",  # Imperial units
    "BTU", "MMBTU", "thm",  # Energy units
    "psi", "atm",  # Pressure
    "gal", "bbl", "scf", "Nm3",  # Volume
    "short_ton", "long_ton",  # Mass
    "CO2", "CO2e", "CH4", "N2O", "SF6",  # GHG
    "kgCO2e", "tCO2e", "lbCO2e", "MtCO2e",  # Emissions (prefix is part of unit)
}


# =============================================================================
# Base Dimensions
# =============================================================================

# Mapping from base units to their dimension
BASE_UNIT_DIMENSIONS: Dict[str, Dict[str, int]] = {
    # SI base units
    "m": {"length": 1},
    "g": {"mass": 1},
    "s": {"time": 1},
    "A": {"current": 1},
    "K": {"temperature": 1},
    "mol": {"amount": 1},
    "cd": {"luminosity": 1},
    # Mass
    "kg": {"mass": 1},
    "t": {"mass": 1},
    "tonne": {"mass": 1},
    "lb": {"mass": 1},
    "oz": {"mass": 1},
    "short_ton": {"mass": 1},
    "long_ton": {"mass": 1},
    # Length
    "ft": {"length": 1},
    "in": {"length": 1},
    "yd": {"length": 1},
    "mi": {"length": 1},
    "mile": {"length": 1},
    # Area
    "ha": {"length": 2},
    "acre": {"length": 2},
    # Volume
    "L": {"length": 3},
    "l": {"length": 3},
    "gal": {"length": 3},
    "bbl": {"length": 3},
    "scf": {"length": 3},
    "Nm3": {"length": 3},
    # Time
    "min": {"time": 1},
    "h": {"time": 1},
    "d": {"time": 1},
    "a": {"time": 1},  # year
    # Energy
    "J": {"mass": 1, "length": 2, "time": -2},
    "Wh": {"mass": 1, "length": 2, "time": -2},
    "kWh": {"mass": 1, "length": 2, "time": -2},
    "MWh": {"mass": 1, "length": 2, "time": -2},
    "BTU": {"mass": 1, "length": 2, "time": -2},
    "MMBTU": {"mass": 1, "length": 2, "time": -2},
    "thm": {"mass": 1, "length": 2, "time": -2},
    # Power
    "W": {"mass": 1, "length": 2, "time": -3},
    "hp": {"mass": 1, "length": 2, "time": -3},
    # Force
    "N": {"mass": 1, "length": 1, "time": -2},
    # Pressure
    "Pa": {"mass": 1, "length": -1, "time": -2},
    "bar": {"mass": 1, "length": -1, "time": -2},
    "psi": {"mass": 1, "length": -1, "time": -2},
    "atm": {"mass": 1, "length": -1, "time": -2},
    # Frequency
    "Hz": {"time": -1},
    # Temperature (special handling for affine)
    "degC": {"temperature": 1},
    "degF": {"temperature": 1},
    # Emissions (treated as mass for dimensional analysis)
    "CO2": {"mass": 1},
    "CO2e": {"mass": 1},
    "kgCO2e": {"mass": 1},
    "tCO2e": {"mass": 1},
    "lbCO2e": {"mass": 1},
    "MtCO2e": {"mass": 1},
    # Dimensionless
    "%": {},
    "ppm": {},
    "ppb": {},
    "ratio": {},
}


@dataclass
class UnitTerm:
    """
    Represents a single term in a unit expression.

    Attributes:
        base: The base unit symbol (e.g., "m", "kg", "s")
        exponent: The exponent applied to the unit (default 1)
        prefix: SI prefix if present (e.g., "k" for kilo)
        prefix_factor: Numeric factor for the prefix

    Example:
        >>> term = UnitTerm(base="m", exponent=2, prefix="k", prefix_factor=Decimal("1000"))
        >>> print(term)  # km**2
    """

    base: str
    exponent: int = 1
    prefix: Optional[str] = None
    prefix_factor: Decimal = field(default_factory=lambda: Decimal("1"))

    def __str__(self) -> str:
        """Return string representation of the term."""
        prefix_str = self.prefix if self.prefix else ""
        exp_str = f"**{self.exponent}" if self.exponent != 1 else ""
        return f"{prefix_str}{self.base}{exp_str}"

    @property
    def full_unit(self) -> str:
        """Return the full unit string including prefix."""
        return f"{self.prefix or ''}{self.base}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "base": self.base,
            "exponent": self.exponent,
            "prefix": self.prefix,
            "prefix_factor": str(self.prefix_factor),
        }


@dataclass
class UnitAST:
    """
    Abstract Syntax Tree for a unit expression.

    Represents a parsed unit as a ratio of products:
    (numerator_terms) / (denominator_terms)

    Attributes:
        raw: Original unit string
        normalized: Normalized unit string
        numerator_terms: List of terms in the numerator
        denominator_terms: List of terms in the denominator
        is_compound: Whether this is a compound unit (has /)
        is_dimensionless: Whether the unit is dimensionless

    Example:
        >>> ast = UnitAST(
        ...     raw="kg*m/s**2",
        ...     numerator_terms=[UnitTerm("kg"), UnitTerm("m")],
        ...     denominator_terms=[UnitTerm("s", exponent=2)]
        ... )
        >>> print(ast.dimension_signature)
        >>> # {"mass": 1, "length": 1, "time": -2}
    """

    raw: str
    normalized: str = ""
    numerator_terms: List[UnitTerm] = field(default_factory=list)
    denominator_terms: List[UnitTerm] = field(default_factory=list)
    is_compound: bool = False
    is_dimensionless: bool = False
    _dimension_cache: Optional[Dict[str, int]] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Set compound flag based on terms."""
        if not self.normalized:
            self.normalized = self._compute_normalized()
        self.is_compound = len(self.denominator_terms) > 0 or len(self.numerator_terms) > 1

    def _compute_normalized(self) -> str:
        """Compute the normalized unit string from terms."""
        if not self.numerator_terms and not self.denominator_terms:
            return "1" if self.is_dimensionless else ""

        num_parts = [str(t) for t in self.numerator_terms]
        num_str = "*".join(num_parts) if num_parts else "1"

        if self.denominator_terms:
            den_parts = [str(t) for t in self.denominator_terms]
            den_str = "*".join(den_parts)
            return f"{num_str}/{den_str}"

        return num_str

    @property
    def dimension_signature(self) -> Dict[str, int]:
        """
        Compute the dimension signature for this unit.

        Returns a dictionary mapping dimension names to their exponents.
        Numerator terms add to exponents, denominator terms subtract.

        Returns:
            Dictionary of dimension names to exponents

        Example:
            >>> ast = parse_unit_to_ast("kg*m/s**2")
            >>> sig = ast.dimension_signature
            >>> assert sig == {"mass": 1, "length": 1, "time": -2}
        """
        if self._dimension_cache is not None:
            return self._dimension_cache

        dimensions: Dict[str, int] = {}

        # Process numerator terms (positive exponents)
        for term in self.numerator_terms:
            term_dims = self._get_term_dimensions(term)
            for dim, exp in term_dims.items():
                dimensions[dim] = dimensions.get(dim, 0) + exp * term.exponent

        # Process denominator terms (negative exponents)
        for term in self.denominator_terms:
            term_dims = self._get_term_dimensions(term)
            for dim, exp in term_dims.items():
                dimensions[dim] = dimensions.get(dim, 0) - exp * term.exponent

        # Remove zero exponents
        dimensions = {k: v for k, v in dimensions.items() if v != 0}

        self._dimension_cache = dimensions
        return dimensions

    def _get_term_dimensions(self, term: UnitTerm) -> Dict[str, int]:
        """Get the base dimensions for a unit term."""
        # Try full unit first (with prefix)
        full_unit = term.full_unit
        if full_unit in BASE_UNIT_DIMENSIONS:
            return BASE_UNIT_DIMENSIONS[full_unit].copy()

        # Try base unit
        if term.base in BASE_UNIT_DIMENSIONS:
            return BASE_UNIT_DIMENSIONS[term.base].copy()

        # Unknown unit - return empty (will be caught by validation)
        logger.warning(f"Unknown unit for dimension lookup: {term.base}")
        return {}

    @property
    def dimension_string(self) -> str:
        """
        Get a canonical string representation of the dimension.

        Returns:
            Sorted dimension string like "length**1*mass**1*time**-2"
        """
        if not self.dimension_signature:
            return "dimensionless"

        parts = []
        for dim in sorted(self.dimension_signature.keys()):
            exp = self.dimension_signature[dim]
            if exp == 1:
                parts.append(dim)
            else:
                parts.append(f"{dim}**{exp}")

        return "*".join(parts)

    @property
    def provenance_hash(self) -> str:
        """
        Compute SHA-256 hash for audit provenance.

        Returns:
            Hex string of SHA-256 hash
        """
        content = f"{self.raw}|{self.normalized}|{self.dimension_string}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert AST to dictionary representation."""
        return {
            "raw": self.raw,
            "normalized": self.normalized,
            "numerator_terms": [t.to_dict() for t in self.numerator_terms],
            "denominator_terms": [t.to_dict() for t in self.denominator_terms],
            "is_compound": self.is_compound,
            "is_dimensionless": self.is_dimensionless,
            "dimension_signature": self.dimension_signature,
            "dimension_string": self.dimension_string,
        }

    def is_compatible_with(self, other: "UnitAST") -> bool:
        """
        Check if this unit is dimensionally compatible with another.

        Args:
            other: Another UnitAST to compare

        Returns:
            True if dimensions match
        """
        return self.dimension_signature == other.dimension_signature


class UnitASTParser:
    """
    Parser for converting unit strings to AST representation.

    The parser handles:
    - Simple units (m, kg, s)
    - Prefixed units (km, mg, ms)
    - Exponents (m**2, s**-1)
    - Compound units (kg/m**3, m*s**-2)
    - Products (kg*m)

    Example:
        >>> parser = UnitASTParser()
        >>> ast = parser.parse("kg*m/s**2")
        >>> print(ast.dimension_signature)
    """

    # Regex patterns for parsing
    TERM_PATTERN = re.compile(
        r"^"
        r"(?P<prefix>[YZEPTGMkhdcmunpfazy]|da|\u03bc|\u00b5)?"
        r"(?P<base>[a-zA-Z_][a-zA-Z0-9_]*)"
        r"(?:\*\*(?P<exponent>-?\d+))?"
        r"$"
    )

    def __init__(
        self,
        custom_dimensions: Optional[Dict[str, Dict[str, int]]] = None,
        strict_prefix_mode: bool = False,
    ) -> None:
        """
        Initialize the parser.

        Args:
            custom_dimensions: Additional dimension mappings for custom units
            strict_prefix_mode: If True, reject units with invalid prefixes
        """
        self.dimensions = {**BASE_UNIT_DIMENSIONS}
        if custom_dimensions:
            self.dimensions.update(custom_dimensions)

        self.strict_prefix_mode = strict_prefix_mode

    def parse(self, unit_string: str) -> UnitAST:
        """
        Parse a unit string into an AST.

        Args:
            unit_string: Preprocessed unit string to parse

        Returns:
            UnitAST representation

        Raises:
            UnitParseError: If the unit string cannot be parsed
            InvalidExponentError: If an exponent is invalid
            UnsupportedCompoundError: If compound structure is not supported

        Example:
            >>> parser = UnitASTParser()
            >>> ast = parser.parse("kg*m/s**2")
            >>> print(ast.numerator_terms)  # [UnitTerm("kg"), UnitTerm("m")]
        """
        if not unit_string or not unit_string.strip():
            return UnitAST(
                raw=unit_string,
                normalized="1",
                is_dimensionless=True
            )

        unit_string = unit_string.strip()

        # Check for unsupported nested divisions
        if unit_string.count("/") > 1:
            raise UnsupportedCompoundError(
                raw_unit=unit_string,
                reason="Multiple division operators not supported. Use parentheses or single division."
            )

        # Split numerator and denominator
        if "/" in unit_string:
            parts = unit_string.split("/", 1)
            numerator_str = parts[0].strip()
            denominator_str = parts[1].strip()
        else:
            numerator_str = unit_string
            denominator_str = ""

        # Parse numerator terms
        numerator_terms = self._parse_product(numerator_str) if numerator_str else []

        # Parse denominator terms
        denominator_terms = self._parse_product(denominator_str) if denominator_str else []

        # Check for dimensionless
        is_dimensionless = (
            not numerator_terms and
            not denominator_terms
        ) or unit_string in ("1", "dimensionless", "ratio", "%", "ppm", "ppb")

        return UnitAST(
            raw=unit_string,
            numerator_terms=numerator_terms,
            denominator_terms=denominator_terms,
            is_dimensionless=is_dimensionless,
        )

    def _parse_product(self, product_str: str) -> List[UnitTerm]:
        """
        Parse a product of unit terms (e.g., "kg*m*s**-2").

        Args:
            product_str: String containing product of terms

        Returns:
            List of UnitTerm objects
        """
        if not product_str:
            return []

        # Split by multiplication operator
        term_strs = re.split(r"\*(?!\*)", product_str)
        terms = []

        for term_str in term_strs:
            term_str = term_str.strip()
            if not term_str:
                continue

            term = self._parse_term(term_str)
            if term:
                terms.append(term)

        return terms

    def _parse_term(self, term_str: str) -> UnitTerm:
        """
        Parse a single unit term (e.g., "kg", "m**2", "km").

        Args:
            term_str: Single unit term string

        Returns:
            UnitTerm object

        Raises:
            UnitParseError: If term cannot be parsed
            InvalidExponentError: If exponent is invalid
        """
        # Try regex match first
        match = self.TERM_PATTERN.match(term_str)
        if match:
            groups = match.groupdict()
            prefix = groups.get("prefix")
            base = groups.get("base", term_str)
            exp_str = groups.get("exponent")

            # Parse exponent
            exponent = 1
            if exp_str:
                try:
                    exponent = int(exp_str)
                except ValueError:
                    raise InvalidExponentError(
                        exponent=exp_str,
                        unit=base
                    )

            # Determine if prefix should be applied
            prefix_factor = Decimal("1")
            actual_prefix: Optional[str] = None

            if prefix:
                # Check if this unit should not have prefix stripped
                full_unit = f"{prefix}{base}"
                if full_unit in NO_PREFIX_UNITS or full_unit in self.dimensions:
                    # The "prefix" is actually part of the unit
                    base = full_unit
                    actual_prefix = None
                elif base in PREFIXABLE_UNITS or not self.strict_prefix_mode:
                    # Valid prefix
                    actual_prefix = prefix
                    prefix_factor = SI_PREFIXES.get(prefix, Decimal("1"))
                else:
                    # Invalid prefix for this unit
                    if self.strict_prefix_mode:
                        raise InvalidPrefixError(
                            prefix=prefix,
                            unit=base,
                            valid_prefixes=list(SI_PREFIXES.keys())
                        )
                    else:
                        # Treat as part of the unit
                        base = full_unit
                        actual_prefix = None

            return UnitTerm(
                base=base,
                exponent=exponent,
                prefix=actual_prefix,
                prefix_factor=prefix_factor
            )

        # Fallback: try to extract exponent manually
        exp_match = re.search(r"\*\*(-?\d+)$", term_str)
        if exp_match:
            base = term_str[:exp_match.start()]
            try:
                exponent = int(exp_match.group(1))
            except ValueError:
                raise InvalidExponentError(
                    exponent=exp_match.group(1),
                    unit=base
                )
            return UnitTerm(base=base, exponent=exponent)

        # No exponent found
        return UnitTerm(base=term_str, exponent=1)

    def validate_against_registry(
        self,
        ast: UnitAST,
        known_units: Set[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate AST terms against a set of known units.

        Args:
            ast: UnitAST to validate
            known_units: Set of valid unit symbols

        Returns:
            Tuple of (is_valid, list of unknown units)
        """
        unknown = []

        for term in ast.numerator_terms + ast.denominator_terms:
            # Check both full unit and base
            if term.full_unit not in known_units and term.base not in known_units:
                unknown.append(term.full_unit)

        return (len(unknown) == 0, unknown)


def parse_unit_to_ast(unit_string: str) -> UnitAST:
    """
    Convenience function to parse a unit string to AST.

    Args:
        unit_string: Unit string to parse

    Returns:
        UnitAST representation

    Example:
        >>> ast = parse_unit_to_ast("kg*m/s**2")
        >>> print(ast.dimension_signature)
        >>> # {"mass": 1, "length": 1, "time": -2}
    """
    parser = UnitASTParser()
    return parser.parse(unit_string)


def compute_dimension_signature(unit_string: str) -> Dict[str, int]:
    """
    Convenience function to compute dimension signature from unit string.

    Args:
        unit_string: Unit string to analyze

    Returns:
        Dictionary of dimension names to exponents

    Example:
        >>> sig = compute_dimension_signature("kg/m**3")
        >>> assert sig == {"mass": 1, "length": -3}
    """
    ast = parse_unit_to_ast(unit_string)
    return ast.dimension_signature
