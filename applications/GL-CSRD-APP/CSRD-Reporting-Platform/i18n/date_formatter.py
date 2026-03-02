"""
GL-CSRD-APP Date Formatter

Locale-aware date and time formatting for the CSRD Reporting Platform.
Supports short, medium, long, and ISO date styles with locale-specific
month names, day names, fiscal-year labels, and ESRS time-horizon
descriptions.

Supported locales: EN, DE, FR, ES.

Version: 1.1.0
"""

from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Union

from i18n.locale_manager import LOCALE_CONFIGS, LocaleConfig, SUPPORTED_LOCALES

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Month and day names per locale
# -----------------------------------------------------------------------

MONTH_NAMES: Dict[str, List[str]] = {
    "en": [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ],
    "de": [
        "Januar", "Februar", "M\u00e4rz", "April", "Mai", "Juni",
        "Juli", "August", "September", "Oktober", "November", "Dezember",
    ],
    "fr": [
        "janvier", "f\u00e9vrier", "mars", "avril", "mai", "juin",
        "juillet", "ao\u00fbt", "septembre", "octobre", "novembre", "d\u00e9cembre",
    ],
    "es": [
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
    ],
}

MONTH_ABBREVIATIONS: Dict[str, List[str]] = {
    "en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "de": ["Jan", "Feb", "M\u00e4r", "Apr", "Mai", "Jun",
           "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"],
    "fr": ["janv.", "f\u00e9vr.", "mars", "avr.", "mai", "juin",
           "juil.", "ao\u00fbt", "sept.", "oct.", "nov.", "d\u00e9c."],
    "es": ["ene", "feb", "mar", "abr", "may", "jun",
           "jul", "ago", "sep", "oct", "nov", "dic"],
}

DAY_NAMES: Dict[str, List[str]] = {
    "en": ["Monday", "Tuesday", "Wednesday", "Thursday",
           "Friday", "Saturday", "Sunday"],
    "de": ["Montag", "Dienstag", "Mittwoch", "Donnerstag",
           "Freitag", "Samstag", "Sonntag"],
    "fr": ["lundi", "mardi", "mercredi", "jeudi",
           "vendredi", "samedi", "dimanche"],
    "es": ["lunes", "martes", "mi\u00e9rcoles", "jueves",
           "viernes", "s\u00e1bado", "domingo"],
}

DAY_ABBREVIATIONS: Dict[str, List[str]] = {
    "en": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "de": ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"],
    "fr": ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."],
    "es": ["lun", "mar", "mi\u00e9", "jue", "vie", "s\u00e1b", "dom"],
}

# -----------------------------------------------------------------------
# ESRS time-horizon translations
# -----------------------------------------------------------------------

TIME_HORIZONS: Dict[str, Dict[str, str]] = {
    "en": {
        "short": "Short-term (<1 year)",
        "medium": "Medium-term (1-5 years)",
        "long": "Long-term (>5 years)",
    },
    "de": {
        "short": "Kurzfristig (<1 Jahr)",
        "medium": "Mittelfristig (1-5 Jahre)",
        "long": "Langfristig (>5 Jahre)",
    },
    "fr": {
        "short": "Court terme (<1 an)",
        "medium": "Moyen terme (1-5 ans)",
        "long": "Long terme (>5 ans)",
    },
    "es": {
        "short": "Corto plazo (<1 a\u00f1o)",
        "medium": "Medio plazo (1-5 a\u00f1os)",
        "long": "Largo plazo (>5 a\u00f1os)",
    },
}


# -----------------------------------------------------------------------
# DateFormatter
# -----------------------------------------------------------------------
class DateFormatter:
    """Locale-aware date and time formatter.

    Provides methods for formatting dates, datetimes, reporting periods,
    fiscal-year labels, and ESRS time-horizon descriptions with
    locale-specific month/day names.

    Example:
        >>> fmt = DateFormatter()
        >>> fmt.format_date(datetime(2024, 1, 15), "en", style="long")
        'January 15, 2024'
        >>> fmt.format_date(datetime(2024, 1, 15), "de", style="long")
        '15. Januar 2024'
    """

    def __init__(self) -> None:
        """Initialise the formatter."""
        logger.debug("DateFormatter initialised")

    # ----- Core formatting ---------------------------------------------

    def format_date(
        self,
        dt: Union[datetime, date],
        locale: str = "en",
        style: str = "medium",
    ) -> str:
        """Format *dt* as a locale-specific date string.

        Args:
            dt: A ``datetime`` or ``date`` object.
            locale: Target locale code.
            style: Formatting style:
                - ``"short"``: Numeric date using locale pattern (e.g. ``01/15/2024``).
                - ``"medium"``: Abbreviated month name (e.g. ``Jan 15, 2024``).
                - ``"long"``: Full month name (e.g. ``January 15, 2024``).
                - ``"iso"``: ISO 8601 (``2024-01-15``).

        Returns:
            Formatted date string.

        Raises:
            ValueError: If *locale* or *style* is invalid.
        """
        self._validate_locale(locale)
        config = LOCALE_CONFIGS[locale]

        if style == "iso":
            return self._format_iso_date(dt)
        elif style == "short":
            return self._format_short_date(dt, config)
        elif style == "medium":
            return self._format_medium_date(dt, locale)
        elif style == "long":
            return self._format_long_date(dt, locale, config)
        else:
            raise ValueError(
                f"Unknown date style '{style}'. "
                f"Supported: short, medium, long, iso"
            )

    def format_datetime(
        self,
        dt: datetime,
        locale: str = "en",
    ) -> str:
        """Format *dt* as a full locale-specific datetime string.

        Args:
            dt: Datetime object.
            locale: Target locale code.

        Returns:
            Formatted datetime string (e.g. ``"01/15/2024 02:30 PM"``).
        """
        self._validate_locale(locale)
        config = LOCALE_CONFIGS[locale]

        date_part = self._format_short_date(dt, config)

        if locale == "en":
            hour = dt.hour % 12 or 12
            ampm = "AM" if dt.hour < 12 else "PM"
            time_part = f"{hour:02d}:{dt.minute:02d} {ampm}"
        else:
            time_part = f"{dt.hour:02d}:{dt.minute:02d}"

        return f"{date_part} {time_part}"

    def format_reporting_period(
        self,
        start: Union[datetime, date],
        end: Union[datetime, date],
        locale: str = "en",
    ) -> str:
        """Format a reporting period label.

        If the period spans a full calendar year (Jan 1 to Dec 31), a
        fiscal-year label is returned (e.g. ``"FY2024"``).  Otherwise a
        date-range string is returned.

        Args:
            start: Period start date.
            end: Period end date.
            locale: Target locale code.

        Returns:
            Reporting period label.
        """
        self._validate_locale(locale)
        config = LOCALE_CONFIGS[locale]

        is_full_year = (
            start.month == 1
            and start.day == 1
            and end.month == 12
            and end.day == 31
            and start.year == end.year
        )

        if is_full_year:
            return self.get_fiscal_year_label(start.year, locale)

        # Range format
        start_str = self._format_short_date(start, config)
        end_str = self._format_short_date(end, config)

        range_templates = {
            "en": f"{start_str} - {end_str}",
            "de": f"{start_str} - {end_str}",
            "fr": f"du {start_str} au {end_str}",
            "es": f"del {start_str} al {end_str}",
        }
        return range_templates.get(locale, f"{start_str} - {end_str}")

    def get_fiscal_year_label(
        self,
        year: int,
        locale: str = "en",
    ) -> str:
        """Return a fiscal-year label for *year*.

        EN -> ``FY2024``, DE -> ``GJ2024``, FR -> ``EF2024``, ES -> ``EF2024``.

        Args:
            year: Four-digit year.
            locale: Target locale code.

        Returns:
            Fiscal-year label string.
        """
        self._validate_locale(locale)
        config = LOCALE_CONFIGS[locale]
        return f"{config.fiscal_year_prefix}{year}"

    def format_time_horizon(
        self,
        horizon: str,
        locale: str = "en",
    ) -> str:
        """Return the translated time-horizon description.

        Args:
            horizon: One of ``"short"``, ``"medium"``, ``"long"``.
            locale: Target locale code.

        Returns:
            Translated time-horizon string.

        Raises:
            ValueError: If *horizon* is not recognised.
        """
        self._validate_locale(locale)
        horizons = TIME_HORIZONS.get(locale, TIME_HORIZONS["en"])
        if horizon not in horizons:
            raise ValueError(
                f"Unknown time horizon '{horizon}'. "
                f"Supported: short, medium, long"
            )
        return horizons[horizon]

    # ----- Parsing ------------------------------------------------------

    def parse_date(
        self,
        text: str,
        locale: str = "en",
    ) -> datetime:
        """Parse a locale-formatted date string into a ``datetime``.

        Supports the short-date pattern for each locale.

        Args:
            text: Date string to parse.
            locale: Source locale code.

        Returns:
            Parsed ``datetime`` object.

        Raises:
            ValueError: If the text cannot be parsed.
        """
        self._validate_locale(locale)
        config = LOCALE_CONFIGS[locale]
        text = text.strip()

        # Try ISO format first
        try:
            return datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            pass

        # Try locale short format
        try:
            return datetime.strptime(text, config.date_format)
        except ValueError:
            pass

        raise ValueError(
            f"Cannot parse '{text}' as a date (locale={locale}). "
            f"Expected format: {config.date_format} or YYYY-MM-DD"
        )

    # ----- Accessors for month/day names --------------------------------

    def get_month_name(
        self,
        month: int,
        locale: str = "en",
        abbreviated: bool = False,
    ) -> str:
        """Return the name of month *month* (1-12) in *locale*.

        Args:
            month: Month number (1 = January).
            locale: Target locale code.
            abbreviated: If ``True``, return the abbreviated form.

        Returns:
            Month name string.

        Raises:
            ValueError: If *month* is out of range.
        """
        if not 1 <= month <= 12:
            raise ValueError(f"Month must be 1-12, got {month}")
        self._validate_locale(locale)
        names = MONTH_ABBREVIATIONS if abbreviated else MONTH_NAMES
        return names.get(locale, names["en"])[month - 1]

    def get_day_name(
        self,
        weekday: int,
        locale: str = "en",
        abbreviated: bool = False,
    ) -> str:
        """Return the name of weekday *weekday* (0=Monday) in *locale*.

        Args:
            weekday: Weekday number (0 = Monday, 6 = Sunday).
            locale: Target locale code.
            abbreviated: If ``True``, return the abbreviated form.

        Returns:
            Day name string.

        Raises:
            ValueError: If *weekday* is out of range.
        """
        if not 0 <= weekday <= 6:
            raise ValueError(f"Weekday must be 0-6, got {weekday}")
        self._validate_locale(locale)
        names = DAY_ABBREVIATIONS if abbreviated else DAY_NAMES
        return names.get(locale, names["en"])[weekday]

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
    def _format_iso_date(dt: Union[datetime, date]) -> str:
        """Return ISO 8601 date string ``YYYY-MM-DD``."""
        return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"

    @staticmethod
    def _format_short_date(
        dt: Union[datetime, date],
        config: LocaleConfig,
    ) -> str:
        """Format date using the locale's short date pattern."""
        if isinstance(dt, datetime):
            return dt.strftime(config.date_format)
        return datetime(dt.year, dt.month, dt.day).strftime(config.date_format)

    @staticmethod
    def _format_medium_date(
        dt: Union[datetime, date],
        locale: str,
    ) -> str:
        """Format date with abbreviated month name.

        EN: ``Jan 15, 2024``
        DE: ``15. Jan 2024``
        FR: ``15 janv. 2024``
        ES: ``15 ene 2024``
        """
        abbr = MONTH_ABBREVIATIONS.get(locale, MONTH_ABBREVIATIONS["en"])
        month_abbr = abbr[dt.month - 1]

        if locale == "en":
            return f"{month_abbr} {dt.day}, {dt.year}"
        elif locale == "de":
            return f"{dt.day}. {month_abbr} {dt.year}"
        elif locale == "fr":
            return f"{dt.day} {month_abbr} {dt.year}"
        elif locale == "es":
            return f"{dt.day} {month_abbr} {dt.year}"
        else:
            return f"{month_abbr} {dt.day}, {dt.year}"

    @staticmethod
    def _format_long_date(
        dt: Union[datetime, date],
        locale: str,
        config: LocaleConfig,
    ) -> str:
        """Format date with full month name.

        EN: ``January 15, 2024``
        DE: ``15. Januar 2024``
        FR: ``15 janvier 2024``
        ES: ``15 de enero de 2024``
        """
        month_name = MONTH_NAMES.get(locale, MONTH_NAMES["en"])[dt.month - 1]

        if locale == "en":
            return f"{month_name} {dt.day}, {dt.year}"
        elif locale == "de":
            return f"{dt.day}. {month_name} {dt.year}"
        elif locale == "fr":
            return f"{dt.day} {month_name} {dt.year}"
        elif locale == "es":
            return f"{dt.day} de {month_name} de {dt.year}"
        else:
            return f"{month_name} {dt.day}, {dt.year}"
