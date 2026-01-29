"""
Unit string preprocessor for GL-FOUND-X-003.

This module provides preprocessing utilities for unit strings, handling:
- Unicode normalization (NFC)
- Whitespace and casing normalization
- Separator canonicalization (per -> /, . -> *, etc.)
- Synonym expansion (lbs -> lb, litre -> L, kilograms -> kg, etc.)
- Exponent notation normalization (m2, m^2, m^2 -> m**2)

The preprocessor is designed to be deterministic: the same input will
always produce the same output for a given configuration.

Example:
    >>> from gl_normalizer_core.parser.preprocessor import UnitPreprocessor
    >>> pp = UnitPreprocessor()
    >>> result = pp.preprocess("kilograms per cubic metre")
    >>> print(result.normalized)  # "kg/m**3"
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Synonym Mappings
# =============================================================================

# Common unit synonyms - maps variations to canonical form
UNIT_SYNONYMS: Dict[str, str] = {
    # Mass
    "lbs": "lb",
    "pound": "lb",
    "pounds": "lb",
    "kilogram": "kg",
    "kilograms": "kg",
    "gram": "g",
    "grams": "g",
    "tonne": "t",
    "tonnes": "t",
    "metric_ton": "t",
    "metric_tons": "t",
    "metric ton": "t",
    "metric tons": "t",
    "metricton": "t",
    "metrictons": "t",
    "ton": "short_ton",  # Default to US short ton (locale can override)
    "tons": "short_ton",
    "ounce": "oz",
    "ounces": "oz",
    # Volume
    "litre": "L",
    "litres": "L",
    "liter": "L",
    "liters": "L",
    "millilitre": "mL",
    "millilitres": "mL",
    "milliliter": "mL",
    "milliliters": "mL",
    "gallon": "gal",  # Ambiguous - locale should resolve
    "gallons": "gal",
    "cubic_meter": "m**3",
    "cubic_metre": "m**3",
    "cubic_meters": "m**3",
    "cubic_metres": "m**3",
    "cubic meter": "m**3",
    "cubic metre": "m**3",
    "cubic meters": "m**3",
    "cubic metres": "m**3",
    "cubicmeter": "m**3",
    "cubicmetre": "m**3",
    "barrel": "bbl",
    "barrels": "bbl",
    # Energy
    "kilowatt_hour": "kWh",
    "kilowatt_hours": "kWh",
    "kilowatt hour": "kWh",
    "kilowatt hours": "kWh",
    "kilowatthour": "kWh",
    "kilowatthours": "kWh",
    "megawatt_hour": "MWh",
    "megawatt_hours": "MWh",
    "megawatt hour": "MWh",
    "megawatt hours": "MWh",
    "megawatthour": "MWh",
    "megawatthours": "MWh",
    "joule": "J",
    "joules": "J",
    "kilojoule": "kJ",
    "kilojoules": "kJ",
    "megajoule": "MJ",
    "megajoules": "MJ",
    "gigajoule": "GJ",
    "gigajoules": "GJ",
    "therm": "thm",
    "therms": "thm",
    "btu": "BTU",
    "british_thermal_unit": "BTU",
    "british thermal unit": "BTU",
    "mmbtu": "MMBTU",
    "mm btu": "MMBTU",
    # Power
    "watt": "W",
    "watts": "W",
    "kilowatt": "kW",
    "kilowatts": "kW",
    "megawatt": "MW",
    "megawatts": "MW",
    "gigawatt": "GW",
    "gigawatts": "GW",
    "horsepower": "hp",
    # Length
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "kilometer": "km",
    "kilometers": "km",
    "kilometre": "km",
    "kilometres": "km",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "foot": "ft",
    "feet": "ft",
    "inch": "in",
    "inches": "in",
    "mile": "mi",
    "miles": "mi",
    "yard": "yd",
    "yards": "yd",
    # Area
    "square_meter": "m**2",
    "square_metre": "m**2",
    "square_meters": "m**2",
    "square_metres": "m**2",
    "square meter": "m**2",
    "square metre": "m**2",
    "sq_m": "m**2",
    "sqm": "m**2",
    "square_foot": "ft**2",
    "square_feet": "ft**2",
    "sq_ft": "ft**2",
    "sqft": "ft**2",
    "hectare": "ha",
    "hectares": "ha",
    "acre": "acre",
    "acres": "acre",
    # Time
    "second": "s",
    "seconds": "s",
    "sec": "s",
    "secs": "s",
    "minute": "min",
    "minutes": "min",
    "hour": "h",
    "hours": "h",
    "hr": "h",
    "hrs": "h",
    "day": "d",
    "days": "d",
    "year": "a",
    "years": "a",
    "yr": "a",
    "yrs": "a",
    # Temperature
    "celsius": "degC",
    "centigrade": "degC",
    "fahrenheit": "degF",
    "kelvin": "K",
    # Emissions
    "kgco2e": "kgCO2e",
    "kg_co2e": "kgCO2e",
    "kg co2e": "kgCO2e",
    "kg co2 eq": "kgCO2e",
    "kg co2 equivalent": "kgCO2e",
    "kgco2eq": "kgCO2e",
    "tco2e": "tCO2e",
    "t_co2e": "tCO2e",
    "t co2e": "tCO2e",
    "t co2 eq": "tCO2e",
    "tonne co2e": "tCO2e",
    "tonnes co2e": "tCO2e",
    "tco2eq": "tCO2e",
    "lbco2e": "lbCO2e",
    "lb_co2e": "lbCO2e",
    "lb co2e": "lbCO2e",
    "mtco2e": "MtCO2e",
    "mt co2e": "MtCO2e",
    "megatonne co2e": "MtCO2e",
    # Standard volume conditions
    "normal_cubic_meter": "Nm3",
    "normal cubic meter": "Nm3",
    "normal_cubic_metre": "Nm3",
    "normal cubic metre": "Nm3",
    "nm3": "Nm3",
    "standard_cubic_foot": "scf",
    "standard cubic foot": "scf",
    "standard_cubic_feet": "scf",
    "standard cubic feet": "scf",
}

# Separator mappings - standardize division/multiplication operators
SEPARATOR_MAPPINGS: Dict[str, str] = {
    " per ": "/",
    " / ": "/",
    "per ": "/",
    " per": "/",
    "per": "/",
    " divided by ": "/",
    " by ": "/",
    # Multiplication
    " times ": "*",
    " * ": "*",
    " x ": "*",
    "\u00b7": "*",  # Middle dot
    "\u22c5": "*",  # Dot operator
    "\u2219": "*",  # Bullet operator
    "\u00d7": "*",  # Multiplication sign
    "\u2022": "*",  # Bullet
    " . ": "*",
    ".": "*",  # Only at word boundaries, handled specially
}

# Unicode superscript to regular digit mapping
SUPERSCRIPT_MAP: Dict[str, str] = {
    "\u2070": "0",
    "\u00b9": "1",
    "\u00b2": "2",
    "\u00b3": "3",
    "\u2074": "4",
    "\u2075": "5",
    "\u2076": "6",
    "\u2077": "7",
    "\u2078": "8",
    "\u2079": "9",
    "\u207a": "+",
    "\u207b": "-",
}

# Unicode subscript to regular digit mapping
SUBSCRIPT_MAP: Dict[str, str] = {
    "\u2080": "0",
    "\u2081": "1",
    "\u2082": "2",
    "\u2083": "3",
    "\u2084": "4",
    "\u2085": "5",
    "\u2086": "6",
    "\u2087": "7",
    "\u2088": "8",
    "\u2089": "9",
}


@dataclass
class PreprocessResult:
    """
    Result of unit string preprocessing.

    Attributes:
        original: The original input string
        normalized: The fully normalized unit string
        transformations: List of transformations applied
        warnings: Any warnings generated during preprocessing
    """

    original: str
    normalized: str
    transformations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_transformation(self, name: str, before: str, after: str) -> None:
        """Record a transformation that was applied."""
        if before != after:
            self.transformations.append(f"{name}: '{before}' -> '{after}'")


class UnitPreprocessor:
    """
    Preprocessor for normalizing unit strings before parsing.

    The preprocessor applies a series of deterministic transformations
    to standardize unit strings for parsing. Transformations are applied
    in a specific order to ensure consistent results.

    Attributes:
        synonyms: Dictionary of unit synonyms to expand
        preserve_case_units: Set of units where case should be preserved

    Example:
        >>> pp = UnitPreprocessor()
        >>> result = pp.preprocess("kilograms per cubic metre")
        >>> print(result.normalized)  # "kg/m**3"
    """

    def __init__(
        self,
        synonyms: Optional[Dict[str, str]] = None,
        preserve_case_units: Optional[set] = None,
    ) -> None:
        """
        Initialize the preprocessor.

        Args:
            synonyms: Custom synonym mappings (merged with defaults)
            preserve_case_units: Units where case should be preserved
        """
        self.synonyms = {**UNIT_SYNONYMS}
        if synonyms:
            self.synonyms.update(synonyms)

        # Units where case matters (e.g., mW vs MW)
        self.preserve_case_units = preserve_case_units or {
            "kWh", "MWh", "GWh", "TWh",
            "kW", "MW", "GW", "TW",
            "kJ", "MJ", "GJ", "TJ",
            "mL", "L", "kL", "ML",
            "BTU", "MMBTU",
            "Nm3", "scf",
            "kgCO2e", "tCO2e", "lbCO2e", "MtCO2e",
            "CO2", "CO2e", "CH4", "N2O", "SF6",
            "degC", "degF", "K",
        }

        # Build case-insensitive synonym lookup
        self._synonym_lookup: Dict[str, str] = {}
        for k, v in self.synonyms.items():
            self._synonym_lookup[k.lower()] = v

    def preprocess(self, unit_string: str) -> PreprocessResult:
        """
        Apply all preprocessing steps to a unit string.

        Steps applied in order:
        1. Unicode normalization (NFC)
        2. Whitespace normalization
        3. Unicode superscript/subscript conversion
        4. Separator canonicalization
        5. Synonym expansion
        6. Exponent normalization

        Args:
            unit_string: Raw unit string to preprocess

        Returns:
            PreprocessResult with normalized string and transformation history

        Example:
            >>> pp = UnitPreprocessor()
            >>> result = pp.preprocess("kilograms per m\u00b2")
            >>> print(result.normalized)  # "kg/m**2"
        """
        result = PreprocessResult(original=unit_string, normalized=unit_string)

        if not unit_string or not unit_string.strip():
            result.normalized = ""
            return result

        current = unit_string

        # Step 1: Unicode normalization
        step_before = current
        current = self._normalize_unicode(current)
        result.add_transformation("unicode_nfc", step_before, current)

        # Step 2: Whitespace normalization
        step_before = current
        current = self._normalize_whitespace(current)
        result.add_transformation("whitespace", step_before, current)

        # Step 3: Convert superscripts and subscripts
        step_before = current
        current = self._convert_superscripts(current)
        result.add_transformation("superscripts", step_before, current)

        # Step 4: Canonicalize separators
        step_before = current
        current = self._canonicalize_separators(current)
        result.add_transformation("separators", step_before, current)

        # Step 5: Expand synonyms
        step_before = current
        current = self._expand_synonyms(current)
        result.add_transformation("synonyms", step_before, current)

        # Step 6: Normalize exponents
        step_before = current
        current = self._normalize_exponents(current)
        result.add_transformation("exponents", step_before, current)

        # Step 7: Final cleanup
        step_before = current
        current = self._final_cleanup(current)
        result.add_transformation("cleanup", step_before, current)

        result.normalized = current
        return result

    def _normalize_unicode(self, s: str) -> str:
        """
        Apply Unicode NFC normalization.

        This ensures composed characters are used consistently.

        Args:
            s: Input string

        Returns:
            NFC-normalized string
        """
        return unicodedata.normalize("NFC", s)

    def _normalize_whitespace(self, s: str) -> str:
        """
        Normalize whitespace characters.

        - Convert all whitespace to single spaces
        - Collapse multiple spaces
        - Trim leading/trailing whitespace

        Args:
            s: Input string

        Returns:
            Whitespace-normalized string
        """
        # Replace various whitespace characters with space
        s = re.sub(r"[\t\n\r\f\v]+", " ", s)
        # Collapse multiple spaces
        s = re.sub(r" +", " ", s)
        # Trim
        s = s.strip()
        return s

    def _convert_superscripts(self, s: str) -> str:
        """
        Convert Unicode superscripts to exponent notation.

        Converts characters like m^2 to m**2.

        Args:
            s: Input string

        Returns:
            String with superscripts converted to **n notation
        """
        result = []
        i = 0
        while i < len(s):
            char = s[i]

            # Check for superscript
            if char in SUPERSCRIPT_MAP:
                # Collect all consecutive superscript characters
                exp_digits = []
                while i < len(s) and s[i] in SUPERSCRIPT_MAP:
                    exp_digits.append(SUPERSCRIPT_MAP[s[i]])
                    i += 1
                exp_str = "".join(exp_digits)
                result.append(f"**{exp_str}")
                continue

            # Check for subscript (for chemical formulas like CO2)
            if char in SUBSCRIPT_MAP:
                # Collect all consecutive subscript characters
                sub_digits = []
                while i < len(s) and s[i] in SUBSCRIPT_MAP:
                    sub_digits.append(SUBSCRIPT_MAP[s[i]])
                    i += 1
                sub_str = "".join(sub_digits)
                result.append(sub_str)
                continue

            result.append(char)
            i += 1

        return "".join(result)

    def _canonicalize_separators(self, s: str) -> str:
        """
        Canonicalize division and multiplication separators.

        Converts:
        - "per", " / ", etc. -> "/"
        - "times", " * ", middle dot, etc. -> "*"

        Args:
            s: Input string

        Returns:
            String with canonical separators
        """
        result = s

        # Apply separator mappings (longer patterns first)
        sorted_patterns = sorted(SEPARATOR_MAPPINGS.keys(), key=len, reverse=True)
        for pattern in sorted_patterns:
            replacement = SEPARATOR_MAPPINGS[pattern]
            # Special handling for dot - only at word boundaries
            if pattern == ".":
                # Don't replace dots in numbers or abbreviations
                result = re.sub(
                    r"(?<=[a-zA-Z])\s*\.\s*(?=[a-zA-Z])",
                    replacement,
                    result
                )
            else:
                result = result.replace(pattern, replacement)

        # Clean up spacing around operators
        result = re.sub(r"\s*/\s*", "/", result)
        result = re.sub(r"\s*\*\s*", "*", result)

        return result

    def _expand_synonyms(self, s: str) -> str:
        """
        Expand unit synonyms to canonical forms.

        Handles multi-word synonyms and preserves case where needed.

        Args:
            s: Input string

        Returns:
            String with synonyms expanded
        """
        # First, try to match the entire string
        s_lower = s.lower()
        if s_lower in self._synonym_lookup:
            return self._synonym_lookup[s_lower]

        # Split by operators and process each token
        tokens = re.split(r"([/*])", s)
        result_tokens = []

        for token in tokens:
            token = token.strip()
            if not token:
                continue

            if token in "/*":
                result_tokens.append(token)
                continue

            # Try synonym lookup
            token_lower = token.lower()
            if token_lower in self._synonym_lookup:
                result_tokens.append(self._synonym_lookup[token_lower])
            else:
                # Check if this is a multi-word phrase
                phrase_match = self._try_multi_word_synonym(token)
                if phrase_match:
                    result_tokens.append(phrase_match)
                else:
                    result_tokens.append(token)

        return "".join(result_tokens)

    def _try_multi_word_synonym(self, phrase: str) -> Optional[str]:
        """
        Try to match a multi-word phrase against synonyms.

        Args:
            phrase: Multi-word phrase to match

        Returns:
            Canonical form if matched, None otherwise
        """
        phrase_lower = phrase.lower()
        # Try exact match
        if phrase_lower in self._synonym_lookup:
            return self._synonym_lookup[phrase_lower]

        # Try with underscores replaced by spaces
        phrase_underscore = phrase_lower.replace(" ", "_")
        if phrase_underscore in self._synonym_lookup:
            return self._synonym_lookup[phrase_underscore]

        # Try with spaces replaced by underscores
        phrase_spaced = phrase_lower.replace("_", " ")
        if phrase_spaced in self._synonym_lookup:
            return self._synonym_lookup[phrase_spaced]

        return None

    def _normalize_exponents(self, s: str) -> str:
        """
        Normalize exponent notation to Python-style **.

        Converts:
        - m2, m3 -> m**2, m**3 (for common units)
        - m^2, m^3 -> m**2, m**3
        - m**2 -> m**2 (already correct)

        Args:
            s: Input string

        Returns:
            String with normalized exponents
        """
        # Convert ^n to **n
        s = re.sub(r"\^(-?\d+)", r"**\1", s)

        # Convert naked exponents for common units (m2 -> m**2)
        # Only for known base units to avoid false positives
        base_units = [
            "m", "cm", "mm", "km", "ft", "in", "yd", "mi",  # Length
            "s", "min", "h", "d",  # Time
            "g", "kg", "lb", "t",  # Mass (be careful with t2 etc.)
            "L", "l",  # Volume
            "K",  # Temperature
        ]

        for unit in base_units:
            # Match unit followed by digit(s) that isn't already **
            pattern = rf"(?<!\*)({re.escape(unit)})(\d+)(?!\d)"
            s = re.sub(pattern, r"\1**\2", s)

        return s

    def _final_cleanup(self, s: str) -> str:
        """
        Apply final cleanup to the normalized string.

        - Remove redundant **1 exponents
        - Clean up spacing
        - Validate operator placement

        Args:
            s: Input string

        Returns:
            Cleaned string
        """
        # Remove **1 (equivalent to no exponent)
        s = re.sub(r"\*\*1(?![0-9])", "", s)

        # Clean up double operators
        s = re.sub(r"\*{3,}", "**", s)
        s = re.sub(r"/+", "/", s)

        # Remove leading/trailing operators
        s = re.sub(r"^[/*]+", "", s)
        s = re.sub(r"[/*]+$", "", s)

        return s.strip()

    def expand_synonyms_list(self, unit_string: str) -> Tuple[str, List[str]]:
        """
        Expand synonyms and return both result and list of expansions.

        Args:
            unit_string: Unit string to process

        Returns:
            Tuple of (expanded string, list of expansions made)

        Example:
            >>> pp = UnitPreprocessor()
            >>> result, expansions = pp.expand_synonyms_list("kilograms per litre")
            >>> print(result)  # "kg/L"
            >>> print(expansions)  # ["kilograms -> kg", "litre -> L"]
        """
        expansions: List[str] = []
        tokens = re.split(r"([/*])", unit_string)
        result_tokens = []

        for token in tokens:
            token = token.strip()
            if not token or token in "/*":
                result_tokens.append(token)
                continue

            token_lower = token.lower()
            if token_lower in self._synonym_lookup:
                canonical = self._synonym_lookup[token_lower]
                expansions.append(f"{token} -> {canonical}")
                result_tokens.append(canonical)
            else:
                result_tokens.append(token)

        return "".join(result_tokens), expansions

    def get_all_synonyms(self) -> Dict[str, List[str]]:
        """
        Get a mapping from canonical units to all their synonyms.

        Returns:
            Dictionary mapping canonical unit to list of synonyms

        Example:
            >>> pp = UnitPreprocessor()
            >>> syns = pp.get_all_synonyms()
            >>> print(syns["kg"])  # ["kilogram", "kilograms"]
        """
        canonical_to_synonyms: Dict[str, List[str]] = {}
        for synonym, canonical in self.synonyms.items():
            if canonical not in canonical_to_synonyms:
                canonical_to_synonyms[canonical] = []
            canonical_to_synonyms[canonical].append(synonym)

        # Sort synonym lists for determinism
        for canonical in canonical_to_synonyms:
            canonical_to_synonyms[canonical].sort()

        return canonical_to_synonyms


def preprocess_unit(unit_string: str) -> PreprocessResult:
    """
    Convenience function to preprocess a unit string with default settings.

    Args:
        unit_string: Raw unit string to preprocess

    Returns:
        PreprocessResult with normalized string and transformation history

    Example:
        >>> result = preprocess_unit("kilograms per cubic metre")
        >>> print(result.normalized)  # "kg/m**3"
    """
    preprocessor = UnitPreprocessor()
    return preprocessor.preprocess(unit_string)
