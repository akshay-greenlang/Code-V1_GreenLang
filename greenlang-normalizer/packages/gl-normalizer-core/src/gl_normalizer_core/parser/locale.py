"""
Locale handler for GL-FOUND-X-003 Unit & Reference Normalizer.

This module provides locale-aware handling for:
- Ambiguous unit resolution (gallon -> gal_us or gal_uk, ton -> short_ton or metric_tonne)
- Numeric string parsing for CSV inputs (1,234 vs 1.234)
- Resolution order: request -> dataset -> org

Example:
    >>> from gl_normalizer_core.parser.locale import LocaleHandler, LocaleProfile
    >>> handler = LocaleHandler()
    >>> profile = LocaleProfile(locale_code="en_US")
    >>> resolved = handler.resolve_ambiguous_unit("gallon", profile)
    >>> print(resolved)  # "gal_us"
"""

import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from gl_normalizer_core.parser.exceptions import (
    AmbiguousUnitError,
    LocaleRequiredError,
)

logger = logging.getLogger(__name__)


class LocaleResolutionLevel(Enum):
    """Resolution level for locale settings."""

    REQUEST = "request"  # Highest priority - per-request override
    DATASET = "dataset"  # Dataset-level default
    ORG = "org"          # Organization-level default
    SYSTEM = "system"    # System-wide fallback


# =============================================================================
# Locale-Specific Unit Mappings
# =============================================================================

# Ambiguous units and their locale-specific resolutions
AMBIGUOUS_UNITS: Dict[str, Dict[str, str]] = {
    # Gallon - US vs Imperial
    "gallon": {
        "en_US": "gal_us",
        "en_GB": "gal_uk",
        "en_AU": "gal_uk",
        "en_CA": "gal_us",  # Canada uses US gallon for fuel
        "default": "gal_us",  # Default to US
    },
    "gal": {
        "en_US": "gal_us",
        "en_GB": "gal_uk",
        "en_AU": "gal_uk",
        "en_CA": "gal_us",
        "default": "gal_us",
    },
    # Ton - Short (US) vs Long (UK) vs Metric
    "ton": {
        "en_US": "short_ton",
        "en_GB": "long_ton",
        "en_AU": "metric_tonne",
        "en_CA": "short_ton",
        "default": "metric_tonne",  # Default to metric for international
    },
    "tons": {
        "en_US": "short_ton",
        "en_GB": "long_ton",
        "en_AU": "metric_tonne",
        "en_CA": "short_ton",
        "default": "metric_tonne",
    },
    # Ounce - Fluid vs Weight (context-dependent)
    "oz": {
        "fluid_context": "fl_oz",
        "mass_context": "oz",
        "default": "oz",  # Default to mass
    },
    "ounce": {
        "fluid_context": "fl_oz",
        "mass_context": "oz",
        "default": "oz",
    },
    # Cup - US vs Metric
    "cup": {
        "en_US": "cup_us",
        "en_GB": "cup_metric",
        "en_AU": "cup_metric",
        "default": "cup_us",
    },
    # Pint - US vs Imperial
    "pint": {
        "en_US": "pint_us",
        "en_GB": "pint_uk",
        "en_AU": "pint_uk",
        "default": "pint_us",
    },
    "pt": {
        "en_US": "pint_us",
        "en_GB": "pint_uk",
        "en_AU": "pint_uk",
        "default": "pint_us",
    },
    # Quart - US vs Imperial
    "quart": {
        "en_US": "quart_us",
        "en_GB": "quart_uk",
        "en_AU": "quart_uk",
        "default": "quart_us",
    },
    "qt": {
        "en_US": "quart_us",
        "en_GB": "quart_uk",
        "en_AU": "quart_uk",
        "default": "quart_us",
    },
    # Barrel - Oil vs Beer vs Other
    "barrel": {
        "oil_context": "bbl_oil",
        "beer_context": "bbl_beer",
        "default": "bbl_oil",  # Default to oil barrel in energy context
    },
    "bbl": {
        "oil_context": "bbl_oil",
        "beer_context": "bbl_beer",
        "default": "bbl_oil",
    },
    # Hundredweight - Short vs Long
    "cwt": {
        "en_US": "cwt_short",
        "en_GB": "cwt_long",
        "default": "cwt_short",
    },
    "hundredweight": {
        "en_US": "cwt_short",
        "en_GB": "cwt_long",
        "default": "cwt_short",
    },
    # BTU - ISO vs IT vs thermochemical
    "btu": {
        "default": "BTU_IT",  # International Table BTU
        "thermochemical": "BTU_th",
        "ISO": "BTU_ISO",
    },
    "BTU": {
        "default": "BTU_IT",
        "thermochemical": "BTU_th",
        "ISO": "BTU_ISO",
    },
}

# Locale-specific decimal separators
DECIMAL_SEPARATORS: Dict[str, Tuple[str, str]] = {
    # (decimal_char, thousands_char)
    "en_US": (".", ","),
    "en_GB": (".", ","),
    "en_AU": (".", ","),
    "en_CA": (".", ","),
    "de_DE": (",", "."),
    "de_AT": (",", "."),
    "de_CH": (".", "'"),
    "fr_FR": (",", " "),
    "fr_CA": (",", " "),
    "es_ES": (",", "."),
    "it_IT": (",", "."),
    "nl_NL": (",", "."),
    "pt_BR": (",", "."),
    "pt_PT": (",", "."),
    "pl_PL": (",", " "),
    "ru_RU": (",", " "),
    "ja_JP": (".", ","),
    "zh_CN": (".", ","),
    "ko_KR": (".", ","),
    "default": (".", ","),
}

# Supported locale codes
SUPPORTED_LOCALES: List[str] = list(DECIMAL_SEPARATORS.keys())


@dataclass
class LocaleProfile:
    """
    Locale configuration profile.

    Attributes:
        locale_code: Primary locale identifier (e.g., "en_US", "de_DE")
        fallback_locale: Fallback locale if primary doesn't resolve
        level: Resolution level (request, dataset, org, system)
        context_hints: Additional context for disambiguation (e.g., "oil_context")
        decimal_separator: Override for decimal separator
        thousands_separator: Override for thousands separator

    Example:
        >>> profile = LocaleProfile(
        ...     locale_code="en_US",
        ...     context_hints={"sector": "energy"}
        ... )
    """

    locale_code: str = "en_US"
    fallback_locale: str = "default"
    level: LocaleResolutionLevel = LocaleResolutionLevel.REQUEST
    context_hints: Dict[str, str] = field(default_factory=dict)
    decimal_separator: Optional[str] = None
    thousands_separator: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate and normalize locale code."""
        # Normalize locale code format
        if "_" not in self.locale_code and "-" in self.locale_code:
            self.locale_code = self.locale_code.replace("-", "_")

        # Set default separators from locale if not overridden
        if self.decimal_separator is None or self.thousands_separator is None:
            seps = DECIMAL_SEPARATORS.get(
                self.locale_code,
                DECIMAL_SEPARATORS.get(self.fallback_locale, (".", ","))
            )
            if self.decimal_separator is None:
                self.decimal_separator = seps[0]
            if self.thousands_separator is None:
                self.thousands_separator = seps[1]

    @property
    def language(self) -> str:
        """Extract language code from locale."""
        return self.locale_code.split("_")[0]

    @property
    def region(self) -> Optional[str]:
        """Extract region code from locale."""
        parts = self.locale_code.split("_")
        return parts[1] if len(parts) > 1 else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "locale_code": self.locale_code,
            "fallback_locale": self.fallback_locale,
            "level": self.level.value,
            "context_hints": self.context_hints,
            "decimal_separator": self.decimal_separator,
            "thousands_separator": self.thousands_separator,
        }


@dataclass
class LocaleResolutionResult:
    """
    Result of locale-based resolution.

    Attributes:
        original: Original value before resolution
        resolved: Resolved value
        locale_used: Locale that was used for resolution
        confidence: Confidence in the resolution (1.0 = deterministic)
        warning: Any warning about the resolution
    """

    original: str
    resolved: str
    locale_used: str
    confidence: float = 1.0
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "original": self.original,
            "resolved": self.resolved,
            "locale_used": self.locale_used,
            "confidence": self.confidence,
        }
        if self.warning:
            result["warning"] = self.warning
        return result


class LocaleHandler:
    """
    Handler for locale-aware unit and numeric resolution.

    Provides methods for:
    - Resolving ambiguous units based on locale
    - Parsing numeric strings with locale-specific formatting
    - Managing locale resolution order

    Example:
        >>> handler = LocaleHandler()
        >>> profile = LocaleProfile(locale_code="en_GB")
        >>> result = handler.resolve_ambiguous_unit("gallon", profile)
        >>> print(result.resolved)  # "gal_uk"
    """

    def __init__(
        self,
        default_locale: str = "en_US",
        strict_mode: bool = False,
    ) -> None:
        """
        Initialize the locale handler.

        Args:
            default_locale: Default locale when none specified
            strict_mode: If True, raise errors for ambiguous resolutions
        """
        self.default_locale = default_locale
        self.strict_mode = strict_mode
        self.ambiguous_units = AMBIGUOUS_UNITS.copy()

    def resolve_locale(
        self,
        request_locale: Optional[str] = None,
        dataset_locale: Optional[str] = None,
        org_locale: Optional[str] = None,
    ) -> LocaleProfile:
        """
        Resolve the effective locale from the hierarchy.

        Resolution order: request -> dataset -> org -> system default

        Args:
            request_locale: Per-request locale override
            dataset_locale: Dataset-level locale
            org_locale: Organization-level locale

        Returns:
            LocaleProfile with resolved locale
        """
        # Resolution order
        if request_locale:
            return LocaleProfile(
                locale_code=request_locale,
                level=LocaleResolutionLevel.REQUEST
            )
        if dataset_locale:
            return LocaleProfile(
                locale_code=dataset_locale,
                level=LocaleResolutionLevel.DATASET
            )
        if org_locale:
            return LocaleProfile(
                locale_code=org_locale,
                level=LocaleResolutionLevel.ORG
            )

        return LocaleProfile(
            locale_code=self.default_locale,
            level=LocaleResolutionLevel.SYSTEM
        )

    def resolve_ambiguous_unit(
        self,
        unit: str,
        profile: Optional[LocaleProfile] = None,
        context: Optional[str] = None,
    ) -> LocaleResolutionResult:
        """
        Resolve an ambiguous unit to its locale-specific form.

        Args:
            unit: The ambiguous unit string
            profile: Locale profile (uses default if None)
            context: Additional context (e.g., "oil_context", "fluid_context")

        Returns:
            LocaleResolutionResult with resolved unit

        Raises:
            AmbiguousUnitError: If unit is ambiguous and strict_mode is True
                               without sufficient locale/context information
        """
        profile = profile or LocaleProfile(locale_code=self.default_locale)
        unit_lower = unit.lower()

        # Check if unit is ambiguous
        if unit_lower not in self.ambiguous_units:
            # Not ambiguous, return as-is
            return LocaleResolutionResult(
                original=unit,
                resolved=unit,
                locale_used=profile.locale_code,
                confidence=1.0
            )

        mappings = self.ambiguous_units[unit_lower]

        # Try context first (e.g., "oil_context" for barrel)
        if context and context in mappings:
            return LocaleResolutionResult(
                original=unit,
                resolved=mappings[context],
                locale_used=f"context:{context}",
                confidence=1.0
            )

        # Check context hints in profile
        for hint_key, hint_value in profile.context_hints.items():
            context_key = f"{hint_value}_context"
            if context_key in mappings:
                return LocaleResolutionResult(
                    original=unit,
                    resolved=mappings[context_key],
                    locale_used=f"hint:{hint_key}={hint_value}",
                    confidence=0.95
                )

        # Try locale
        if profile.locale_code in mappings:
            return LocaleResolutionResult(
                original=unit,
                resolved=mappings[profile.locale_code],
                locale_used=profile.locale_code,
                confidence=1.0
            )

        # Try fallback locale
        if profile.fallback_locale in mappings:
            return LocaleResolutionResult(
                original=unit,
                resolved=mappings[profile.fallback_locale],
                locale_used=profile.fallback_locale,
                confidence=0.9,
                warning=f"Used fallback locale {profile.fallback_locale}"
            )

        # Use default mapping
        if "default" in mappings:
            if self.strict_mode:
                # Raise error in strict mode
                candidates = [
                    {"unit": v, "locale": k}
                    for k, v in mappings.items()
                    if k != "default" and not k.endswith("_context")
                ]
                raise AmbiguousUnitError(
                    raw_unit=unit,
                    candidates=candidates,
                    locale_hint=f"Provide locale to disambiguate (tried: {profile.locale_code})"
                )

            return LocaleResolutionResult(
                original=unit,
                resolved=mappings["default"],
                locale_used="default",
                confidence=0.7,
                warning=f"Ambiguous unit '{unit}' resolved using default. "
                        f"Consider specifying locale for accurate resolution."
            )

        # Should not reach here, but return original if we do
        return LocaleResolutionResult(
            original=unit,
            resolved=unit,
            locale_used="none",
            confidence=0.5,
            warning=f"Could not resolve ambiguous unit '{unit}'"
        )

    def get_ambiguous_candidates(self, unit: str) -> List[Dict[str, Any]]:
        """
        Get all possible interpretations of an ambiguous unit.

        Args:
            unit: The unit to check

        Returns:
            List of candidate interpretations with metadata

        Example:
            >>> handler = LocaleHandler()
            >>> candidates = handler.get_ambiguous_candidates("gallon")
            >>> print(candidates)
            >>> # [{"unit": "gal_us", "locale": "en_US"}, ...]
        """
        unit_lower = unit.lower()
        if unit_lower not in self.ambiguous_units:
            return []

        mappings = self.ambiguous_units[unit_lower]
        candidates = []

        for key, value in mappings.items():
            if key == "default":
                continue
            candidate: Dict[str, Any] = {"unit": value}
            if key.endswith("_context"):
                candidate["context"] = key
            else:
                candidate["locale"] = key
            candidates.append(candidate)

        return candidates

    def is_ambiguous(self, unit: str) -> bool:
        """
        Check if a unit is ambiguous and requires locale for resolution.

        Args:
            unit: Unit string to check

        Returns:
            True if unit is in the ambiguous units list
        """
        return unit.lower() in self.ambiguous_units

    def parse_numeric_string(
        self,
        value: str,
        profile: Optional[LocaleProfile] = None,
    ) -> Decimal:
        """
        Parse a numeric string with locale-aware formatting.

        Handles different decimal and thousands separator conventions.

        Args:
            value: Numeric string to parse (e.g., "1,234.56" or "1.234,56")
            profile: Locale profile for separator detection

        Returns:
            Decimal value

        Raises:
            LocaleRequiredError: If format is ambiguous without locale
            ValueError: If string cannot be parsed as a number

        Example:
            >>> handler = LocaleHandler()
            >>> profile = LocaleProfile(locale_code="de_DE")
            >>> result = handler.parse_numeric_string("1.234,56", profile)
            >>> print(result)  # Decimal('1234.56')
        """
        profile = profile or LocaleProfile(locale_code=self.default_locale)
        value = value.strip()

        # Remove any currency symbols or unit suffixes
        value = re.sub(r"[^\d,.\-+eE]", "", value)

        if not value:
            raise ValueError("Empty numeric string")

        # Get separators from profile
        decimal_sep = profile.decimal_separator or "."
        thousands_sep = profile.thousands_separator or ","

        # Check for ambiguous format
        if self._is_ambiguous_numeric(value):
            if self.strict_mode:
                raise LocaleRequiredError(
                    raw_value=value,
                    reason="Numeric format is ambiguous (could be 1,234 or 1.234)",
                    supported_locales=SUPPORTED_LOCALES
                )
            # Use locale hints to resolve
            logger.warning(
                f"Ambiguous numeric format '{value}', using locale {profile.locale_code}"
            )

        # Normalize to standard format (. as decimal)
        if decimal_sep == ",":
            # European format: 1.234,56 -> 1234.56
            value = value.replace(".", "")  # Remove thousands separator
            value = value.replace(",", ".")  # Replace decimal separator
        elif thousands_sep == ".":
            # Just remove thousands separator
            value = value.replace(".", "")
        else:
            # US/UK format: 1,234.56 -> 1234.56
            value = value.replace(",", "")  # Remove thousands separator

        try:
            return Decimal(value)
        except InvalidOperation as e:
            raise ValueError(f"Cannot parse '{value}' as number: {e}")

    def _is_ambiguous_numeric(self, value: str) -> bool:
        """
        Check if a numeric string has ambiguous formatting.

        A format is ambiguous when it could be interpreted either way,
        e.g., "1,234" could be 1234 (US) or 1.234 (DE).

        Args:
            value: Numeric string to check

        Returns:
            True if format is ambiguous
        """
        # Count separators
        dots = value.count(".")
        commas = value.count(",")

        # If both present, check position
        if dots == 1 and commas == 1:
            # If comma comes after dot, ambiguous (could be either format)
            dot_pos = value.rfind(".")
            comma_pos = value.rfind(",")
            # Check if it looks like "1,234.56" (US) or "1.234,56" (EU)
            # Not truly ambiguous if one separator is clearly thousands
            after_last = value[max(dot_pos, comma_pos) + 1:]
            if len(after_last) == 3:
                # Three digits after last separator suggests it's thousands
                return True
            return False

        # Single separator with exactly 3 digits after it is ambiguous
        if dots == 1 and commas == 0:
            after_dot = value[value.rfind(".") + 1:]
            if len(after_dot) == 3 and after_dot.isdigit():
                return True

        if commas == 1 and dots == 0:
            after_comma = value[value.rfind(",") + 1:]
            if len(after_comma) == 3 and after_comma.isdigit():
                return True

        return False

    def get_supported_locales(self) -> List[str]:
        """
        Get list of all supported locale codes.

        Returns:
            List of locale code strings
        """
        return SUPPORTED_LOCALES.copy()


def resolve_unit_locale(
    unit: str,
    locale_code: Optional[str] = None,
    context: Optional[str] = None,
) -> str:
    """
    Convenience function to resolve an ambiguous unit.

    Args:
        unit: Unit string to resolve
        locale_code: Locale code (e.g., "en_US")
        context: Additional context

    Returns:
        Resolved unit string

    Example:
        >>> resolved = resolve_unit_locale("gallon", "en_GB")
        >>> print(resolved)  # "gal_uk"
    """
    handler = LocaleHandler()
    profile = LocaleProfile(locale_code=locale_code or "en_US")
    result = handler.resolve_ambiguous_unit(unit, profile, context)
    return result.resolved


def parse_number(
    value: str,
    locale_code: Optional[str] = None,
) -> Decimal:
    """
    Convenience function to parse a numeric string with locale.

    Args:
        value: Numeric string to parse
        locale_code: Locale code for format detection

    Returns:
        Decimal value

    Example:
        >>> num = parse_number("1.234,56", "de_DE")
        >>> print(num)  # Decimal('1234.56')
    """
    handler = LocaleHandler()
    profile = LocaleProfile(locale_code=locale_code or "en_US")
    return handler.parse_numeric_string(value, profile)
