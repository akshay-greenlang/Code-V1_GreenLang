"""
GL-CSRD-APP Number Formatter

Locale-aware numeric formatting for the CSRD Reporting Platform.  Uses
``Decimal`` arithmetic throughout for regulatory-grade precision and
supports locale-specific thousand/decimal separators, currency symbols,
and scientific/large-number abbreviations.

Supported locales:
    EN: 1,234,567.89
    DE: 1.234.567,89
    FR: 1 234 567,89  (narrow no-break space)
    ES: 1.234.567,89

Version: 1.1.0
"""

from __future__ import annotations

import logging
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Dict, Optional, Set, Tuple, Union

from i18n.locale_manager import LOCALE_CONFIGS, LocaleConfig, SUPPORTED_LOCALES

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

# ISO 4217 currency codes and their symbols
CURRENCY_SYMBOLS: Dict[str, str] = {
    "EUR": "\u20ac",   # Euro sign
    "USD": "$",
    "GBP": "\u00a3",   # Pound sign
    "CHF": "CHF",
    "SEK": "kr",
    "NOK": "kr",
    "DKK": "kr",
    "PLN": "z\u0142",
    "CZK": "K\u010d",
    "HUF": "Ft",
}

VALID_CURRENCY_CODES: Set[str] = set(CURRENCY_SYMBOLS.keys())

# Large-number abbreviation suffixes per locale
_LARGE_NUMBER_SUFFIXES: Dict[str, Dict[str, Tuple[float, str]]] = {
    "en": {
        "T": (1e12, "T"),
        "B": (1e9, "B"),
        "M": (1e6, "M"),
        "K": (1e3, "K"),
    },
    "de": {
        "T": (1e12, " Bio."),
        "B": (1e9, " Mrd."),
        "M": (1e6, " Mio."),
        "K": (1e3, " Tsd."),
    },
    "fr": {
        "T": (1e12, " T"),
        "B": (1e9, " Md"),
        "M": (1e6, " M"),
        "K": (1e3, " k"),
    },
    "es": {
        "T": (1e12, " B"),
        "B": (1e9, " MM"),
        "M": (1e6, " M"),
        "K": (1e3, " k"),
    },
}


# -----------------------------------------------------------------------
# NumberFormatter
# -----------------------------------------------------------------------
class NumberFormatter:
    """Locale-aware number formatter with ``Decimal`` precision.

    All formatting methods accept ``int``, ``float``, or ``Decimal``
    values and return locale-formatted strings.  Internal arithmetic
    uses ``Decimal`` with ``ROUND_HALF_UP`` to comply with regulatory
    rounding requirements.

    Example:
        >>> fmt = NumberFormatter()
        >>> fmt.format_number(1234567.89, "de")
        '1.234.567,89'
        >>> fmt.format_currency(1234.56, "EUR", "fr")
        '1 234,56 \u20ac'
    """

    def __init__(self) -> None:
        """Initialise the formatter.  No configuration required."""
        logger.debug("NumberFormatter initialised")

    # ----- Core formatting ---------------------------------------------

    def format_number(
        self,
        value: Union[int, float, Decimal],
        locale: str = "en",
        decimals: int = 2,
    ) -> str:
        """Format a numeric *value* with locale separators.

        Args:
            value: The number to format.
            locale: Target locale code.
            decimals: Number of decimal places.

        Returns:
            Formatted number string.

        Raises:
            ValueError: If *locale* is not supported or *value* is not
                convertible to ``Decimal``.
        """
        self._validate_locale(locale)
        config = LOCALE_CONFIGS[locale]
        dec = self._to_decimal(value, decimals)
        return self._apply_locale_format(dec, config, decimals)

    def format_percentage(
        self,
        value: Union[int, float, Decimal],
        locale: str = "en",
        decimals: int = 2,
    ) -> str:
        """Format *value* as a percentage string.

        The value is assumed to already be in percentage form (e.g.
        ``45.2`` for 45.2%).

        Args:
            value: Percentage value.
            locale: Target locale code.
            decimals: Decimal places.

        Returns:
            Formatted percentage, e.g. ``"45.20%"`` or ``"45,20%"``.
        """
        formatted = self.format_number(value, locale, decimals=decimals)
        return f"{formatted}%"

    def format_currency(
        self,
        amount: Union[int, float, Decimal],
        currency_code: str,
        locale: str = "en",
        decimals: int = 2,
    ) -> str:
        """Format *amount* as a currency string.

        Args:
            amount: Monetary value.
            currency_code: ISO 4217 code (e.g. ``"EUR"``).
            locale: Target locale code.
            decimals: Decimal places (usually 2).

        Returns:
            Formatted currency string with symbol.

        Raises:
            ValueError: If *currency_code* is not a recognised ISO 4217 code.
        """
        self._validate_currency(currency_code)
        config = LOCALE_CONFIGS[locale]
        symbol = CURRENCY_SYMBOLS.get(currency_code, currency_code)
        formatted_number = self.format_number(amount, locale, decimals=decimals)

        if config.currency_symbol_position == "before":
            return f"{symbol}{formatted_number}"
        else:
            return f"{formatted_number} {symbol}"

    # ----- Domain-specific formatters -----------------------------------

    def format_emissions(
        self,
        value: Union[int, float, Decimal],
        locale: str = "en",
        unit: str = "tCO2e",
        decimals: int = 2,
    ) -> str:
        """Format *value* as an emissions quantity with unit label.

        Args:
            value: Emissions value.
            locale: Target locale.
            unit: Unit label (default ``"tCO2e"``).  Also supports
                ``"kgCO2e"``, ``"MtCO2e"``, etc.
            decimals: Decimal places.

        Returns:
            Formatted string, e.g. ``"1,234.56 tCO2e"``.
        """
        formatted = self.format_number(value, locale, decimals=decimals)
        # Use subscript-style label for display
        display_unit = unit.replace("CO2e", "CO\u2082e")
        return f"{formatted} {display_unit}"

    def format_energy(
        self,
        value: Union[int, float, Decimal],
        locale: str = "en",
        unit: str = "MWh",
        decimals: int = 2,
    ) -> str:
        """Format *value* as an energy quantity with unit.

        Args:
            value: Energy value.
            locale: Target locale.
            unit: Unit label (``"MWh"``, ``"GJ"``, ``"kWh"``, ``"TJ"``).
            decimals: Decimal places.

        Returns:
            Formatted energy string.
        """
        formatted = self.format_number(value, locale, decimals=decimals)
        return f"{formatted} {unit}"

    def format_area(
        self,
        value: Union[int, float, Decimal],
        locale: str = "en",
        unit: str = "ha",
        decimals: int = 2,
    ) -> str:
        """Format *value* as an area quantity with unit.

        Args:
            value: Area value.
            locale: Target locale.
            unit: Unit label (``"ha"``, ``"km\u00b2"``, ``"m\u00b2"``).
            decimals: Decimal places.

        Returns:
            Formatted area string.
        """
        formatted = self.format_number(value, locale, decimals=decimals)
        # Superscript 2 for square units
        display_unit = unit.replace("km2", "km\u00b2").replace("m2", "m\u00b2")
        return f"{formatted} {display_unit}"

    def format_volume(
        self,
        value: Union[int, float, Decimal],
        locale: str = "en",
        unit: str = "m\u00b3",
        decimals: int = 2,
    ) -> str:
        """Format *value* as a volume quantity with unit.

        Args:
            value: Volume value.
            locale: Target locale.
            unit: Unit label (``"m\u00b3"``, ``"L"``, ``"ML"``).
            decimals: Decimal places.

        Returns:
            Formatted volume string.
        """
        formatted = self.format_number(value, locale, decimals=decimals)
        return f"{formatted} {unit}"

    def format_large_number(
        self,
        value: Union[int, float, Decimal],
        locale: str = "en",
        decimals: int = 1,
    ) -> str:
        """Format *value* with abbreviated magnitude suffix.

        Abbreviations are locale-specific:
            EN: 1.2M, 3.4B, 5.6T
            DE: 1,2 Mio., 3,4 Mrd., 5,6 Bio.

        Args:
            value: Numeric value.
            locale: Target locale.
            decimals: Decimal places after abbreviation.

        Returns:
            Abbreviated number string.
        """
        self._validate_locale(locale)
        abs_value = abs(float(value))
        sign = "-" if float(value) < 0 else ""
        suffixes = _LARGE_NUMBER_SUFFIXES.get(locale, _LARGE_NUMBER_SUFFIXES["en"])

        for _label, (threshold, suffix) in sorted(
            suffixes.items(), key=lambda x: x[1][0], reverse=True
        ):
            if abs_value >= threshold:
                reduced = abs_value / threshold
                formatted = self.format_number(reduced, locale, decimals=decimals)
                return f"{sign}{formatted}{suffix}"

        # Below 1,000 -- format normally
        return self.format_number(value, locale, decimals=decimals)

    # ----- Reverse parsing ----------------------------------------------

    def parse_number(
        self,
        text: str,
        locale: str = "en",
    ) -> Decimal:
        """Parse a locale-formatted number string back to ``Decimal``.

        Args:
            text: Locale-formatted number string (e.g. ``"1.234,56"``).
            locale: Source locale code.

        Returns:
            Parsed ``Decimal`` value.

        Raises:
            ValueError: If the text cannot be parsed.
        """
        self._validate_locale(locale)
        config = LOCALE_CONFIGS[locale]

        cleaned = text.strip()
        # Remove currency symbols and unit labels
        for sym in CURRENCY_SYMBOLS.values():
            cleaned = cleaned.replace(sym, "")
        # Remove common unit labels
        for label in ("tCO2e", "tCO\u2082e", "MWh", "GJ", "kWh", "ha",
                       "km\u00b2", "m\u00b2", "m\u00b3", "%"):
            cleaned = cleaned.replace(label, "")
        cleaned = cleaned.strip()

        # Remove thousands separators
        if config.thousands_separator:
            cleaned = cleaned.replace(config.thousands_separator, "")

        # Normalise decimal separator to dot
        if config.decimal_separator != ".":
            cleaned = cleaned.replace(config.decimal_separator, ".")

        # Remove any remaining non-numeric characters except dot and minus
        # (e.g. abbreviated suffixes)
        for suffix_map in _LARGE_NUMBER_SUFFIXES.values():
            for _, (_, sfx) in suffix_map.items():
                cleaned = cleaned.replace(sfx.strip(), "")
        cleaned = cleaned.strip()

        try:
            return Decimal(cleaned)
        except InvalidOperation as exc:
            raise ValueError(
                f"Cannot parse '{text}' as number (locale={locale}): {exc}"
            ) from exc

    # ----- Internal helpers ---------------------------------------------

    @staticmethod
    def _validate_locale(locale: str) -> None:
        """Raise ``ValueError`` if locale is unsupported."""
        if locale not in SUPPORTED_LOCALES:
            raise ValueError(
                f"Unsupported locale '{locale}'. "
                f"Supported: {', '.join(SUPPORTED_LOCALES)}"
            )

    @staticmethod
    def _validate_currency(code: str) -> None:
        """Raise ``ValueError`` if currency code is unknown."""
        if code not in VALID_CURRENCY_CODES:
            raise ValueError(
                f"Unknown currency code '{code}'. "
                f"Supported: {', '.join(sorted(VALID_CURRENCY_CODES))}"
            )

    @staticmethod
    def _to_decimal(
        value: Union[int, float, Decimal],
        decimals: int,
    ) -> Decimal:
        """Convert *value* to ``Decimal`` rounded to *decimals* places.

        Args:
            value: Input value.
            decimals: Target decimal places.

        Returns:
            Rounded ``Decimal``.

        Raises:
            ValueError: If value cannot be converted.
        """
        try:
            if isinstance(value, Decimal):
                dec = value
            elif isinstance(value, float):
                dec = Decimal(str(value))
            else:
                dec = Decimal(value)
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(f"Cannot convert {value!r} to Decimal: {exc}") from exc

        quantizer = Decimal(10) ** -decimals
        return dec.quantize(quantizer, rounding=ROUND_HALF_UP)

    @staticmethod
    def _apply_locale_format(
        dec: Decimal,
        config: LocaleConfig,
        decimals: int,
    ) -> str:
        """Apply locale-specific separators to a ``Decimal``.

        Args:
            dec: Rounded decimal value.
            config: Locale formatting config.
            decimals: Number of decimal places.

        Returns:
            Formatted string with locale-specific separators.
        """
        sign = "-" if dec < 0 else ""
        abs_dec = abs(dec)
        str_value = str(abs_dec)

        if "." in str_value:
            int_part, frac_part = str_value.split(".", 1)
        else:
            int_part = str_value
            frac_part = "0" * decimals

        # Pad or trim fractional part
        frac_part = frac_part.ljust(decimals, "0")[:decimals]

        # Apply thousands grouping
        grouped_int = NumberFormatter._group_thousands(
            int_part, config.thousands_separator
        )

        if decimals > 0:
            return f"{sign}{grouped_int}{config.decimal_separator}{frac_part}"
        else:
            return f"{sign}{grouped_int}"

    @staticmethod
    def _group_thousands(int_str: str, separator: str) -> str:
        """Insert *separator* every three digits from the right.

        Args:
            int_str: Integer part as a string (no sign).
            separator: Thousands separator character.

        Returns:
            Grouped string.
        """
        if len(int_str) <= 3:
            return int_str

        groups = []
        while len(int_str) > 3:
            groups.append(int_str[-3:])
            int_str = int_str[:-3]
        groups.append(int_str)
        groups.reverse()
        return separator.join(groups)
