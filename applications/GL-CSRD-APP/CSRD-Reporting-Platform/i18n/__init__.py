"""
GL-CSRD-APP Internationalization (i18n) Module

Provides locale-aware formatting and translation for the CSRD Reporting
Platform. Supports EN, DE, FR, ES with locale-specific number formatting,
date formatting, and full ESRS data point label translation.

Version: 1.1.0

Architecture:
    - LocaleManager: Thread-safe singleton managing locale state and translation lookup
    - TranslationCatalog: Loads and caches JSON translation files with fallback chains
    - NumberFormatter: Locale-aware number, currency, and unit formatting (Decimal precision)
    - DateFormatter: Locale-aware date/time formatting with ESRS reporting periods

Usage:
    >>> from i18n import t, format_number, format_date, format_currency
    >>> t("data_points.E1-1.label", locale="de")
    'Brutto-Scope-1-THG-Emissionen'
    >>> format_number(1234567.89, locale="de")
    '1.234.567,89'
    >>> format_currency(1234.56, "EUR", locale="fr")
    '1 234,56 EUR'
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional, Union

from i18n.locale_manager import (
    LocaleConfig,
    LocaleManager,
    TranslationCatalog,
)
from i18n.number_formatter import NumberFormatter
from i18n.date_formatter import DateFormatter

__version__ = "1.1.0"
__all__ = [
    "LocaleManager",
    "TranslationCatalog",
    "LocaleConfig",
    "NumberFormatter",
    "DateFormatter",
    "t",
    "format_number",
    "format_date",
    "format_datetime",
    "format_currency",
    "format_percentage",
    "format_emissions",
    "format_energy",
    "format_reporting_period",
    "get_locale_manager",
    "get_number_formatter",
    "get_date_formatter",
    "set_locale",
    "get_locale",
    "SUPPORTED_LOCALES",
    "DEFAULT_LOCALE",
]

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
SUPPORTED_LOCALES = ("en", "de", "fr", "es")
DEFAULT_LOCALE = "en"

# -----------------------------------------------------------------------
# Module-level singleton accessors
# -----------------------------------------------------------------------
_locale_manager: Optional[LocaleManager] = None
_number_formatter: Optional[NumberFormatter] = None
_date_formatter: Optional[DateFormatter] = None


def get_locale_manager() -> LocaleManager:
    """Return the module-level LocaleManager singleton.

    The singleton is created lazily on first access so that import time
    remains negligible.

    Returns:
        The shared LocaleManager instance.
    """
    global _locale_manager
    if _locale_manager is None:
        _locale_manager = LocaleManager()
        logger.info("Initialized global LocaleManager singleton")
    return _locale_manager


def get_number_formatter() -> NumberFormatter:
    """Return the module-level NumberFormatter singleton.

    Returns:
        The shared NumberFormatter instance.
    """
    global _number_formatter
    if _number_formatter is None:
        _number_formatter = NumberFormatter()
        logger.info("Initialized global NumberFormatter singleton")
    return _number_formatter


def get_date_formatter() -> DateFormatter:
    """Return the module-level DateFormatter singleton.

    Returns:
        The shared DateFormatter instance.
    """
    global _date_formatter
    if _date_formatter is None:
        _date_formatter = DateFormatter()
        logger.info("Initialized global DateFormatter singleton")
    return _date_formatter


# -----------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------


def t(key: str, locale: str = DEFAULT_LOCALE, **params: Any) -> str:
    """Translate *key* into the requested *locale*.

    This is the primary entry point for translation lookups.  It delegates
    to ``LocaleManager.get_message`` which supports dot-notation key
    lookup, placeholder interpolation, and a fallback chain
    (requested locale -> ``en`` -> raw key).

    Args:
        key: Dot-notation translation key, e.g. ``"data_points.E1-1.label"``.
        locale: ISO 639-1 locale code (``en``, ``de``, ``fr``, ``es``).
        **params: Keyword arguments interpolated into the translated string
            using ``str.format_map``.

    Returns:
        The translated (and optionally interpolated) string, or the raw key
        if no translation was found.

    Example:
        >>> t("validation.required_field", locale="de")
        'Dieses Feld ist erforderlich'
        >>> t("success.report_generated", locale="en", report_id="RPT-001")
        'Report RPT-001 generated successfully'
    """
    manager = get_locale_manager()
    return manager.get_message(key, locale, **params)


def set_locale(locale: str) -> None:
    """Set the active locale on the global LocaleManager.

    Args:
        locale: ISO 639-1 locale code.

    Raises:
        ValueError: If *locale* is not in ``SUPPORTED_LOCALES``.
    """
    get_locale_manager().set_locale(locale)


def get_locale() -> str:
    """Return the currently active locale code.

    Returns:
        The active ISO 639-1 locale code.
    """
    return get_locale_manager().active_locale


def format_number(
    value: Union[int, float, Decimal],
    locale: str = DEFAULT_LOCALE,
    decimals: int = 2,
) -> str:
    """Format *value* with locale-specific thousand/decimal separators.

    Args:
        value: Numeric value to format.
        locale: Target locale code.
        decimals: Number of decimal places (default 2).

    Returns:
        Formatted string, e.g. ``"1,234.56"`` (EN) or ``"1.234,56"`` (DE).
    """
    return get_number_formatter().format_number(value, locale, decimals=decimals)


def format_percentage(
    value: Union[int, float, Decimal],
    locale: str = DEFAULT_LOCALE,
) -> str:
    """Format *value* as a percentage string.

    Args:
        value: Percentage value (e.g. 45.2 for 45.2%).
        locale: Target locale code.

    Returns:
        Formatted percentage, e.g. ``"45.20%"`` or ``"45,20%"``.
    """
    return get_number_formatter().format_percentage(value, locale)


def format_currency(
    amount: Union[int, float, Decimal],
    currency: str = "EUR",
    locale: str = DEFAULT_LOCALE,
) -> str:
    """Format *amount* as a currency string with ISO 4217 code.

    Args:
        amount: Monetary value.
        currency: ISO 4217 currency code (e.g. ``"EUR"``).
        locale: Target locale code.

    Returns:
        Formatted currency string, e.g. ``"EUR1,234.56"`` or ``"1.234,56 EUR"``.
    """
    return get_number_formatter().format_currency(amount, currency, locale)


def format_emissions(
    value: Union[int, float, Decimal],
    locale: str = DEFAULT_LOCALE,
    unit: str = "tCO2e",
) -> str:
    """Format *value* as an emissions quantity with unit.

    Args:
        value: Emissions value.
        locale: Target locale code.
        unit: Emissions unit label (default ``"tCO2e"``).

    Returns:
        Formatted emissions string, e.g. ``"1,234.56 tCO2e"``.
    """
    return get_number_formatter().format_emissions(value, locale, unit=unit)


def format_energy(
    value: Union[int, float, Decimal],
    locale: str = DEFAULT_LOCALE,
    unit: str = "MWh",
) -> str:
    """Format *value* as an energy quantity with unit.

    Args:
        value: Energy value.
        locale: Target locale code.
        unit: Energy unit label (default ``"MWh"``).

    Returns:
        Formatted energy string, e.g. ``"1,234.56 MWh"``.
    """
    return get_number_formatter().format_energy(value, locale, unit=unit)


def format_date(
    dt: datetime,
    locale: str = DEFAULT_LOCALE,
    style: str = "medium",
) -> str:
    """Format *dt* as a locale-specific date string.

    Args:
        dt: Datetime object.
        locale: Target locale code.
        style: One of ``"short"``, ``"medium"``, ``"long"``, ``"iso"``.

    Returns:
        Formatted date string.
    """
    return get_date_formatter().format_date(dt, locale, style=style)


def format_datetime(
    dt: datetime,
    locale: str = DEFAULT_LOCALE,
) -> str:
    """Format *dt* as a locale-specific datetime string.

    Args:
        dt: Datetime object.
        locale: Target locale code.

    Returns:
        Formatted datetime string.
    """
    return get_date_formatter().format_datetime(dt, locale)


def format_reporting_period(
    start: datetime,
    end: datetime,
    locale: str = DEFAULT_LOCALE,
) -> str:
    """Format a reporting period label.

    Args:
        start: Period start date.
        end: Period end date.
        locale: Target locale code.

    Returns:
        Reporting period label, e.g. ``"FY2024"`` or ``"GJ2024"``.
    """
    return get_date_formatter().format_reporting_period(start, end, locale)
