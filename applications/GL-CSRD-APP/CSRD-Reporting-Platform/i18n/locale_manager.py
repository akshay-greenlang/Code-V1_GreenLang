"""
GL-CSRD-APP Locale Manager

Thread-safe locale management with lazy-loaded JSON translation catalogs,
dot-notation key lookup, placeholder interpolation, pluralisation, and a
robust fallback chain (requested locale -> ``en`` -> raw key).

Classes:
    LocaleConfig  -- per-locale formatting settings (dataclass).
    TranslationCatalog -- loads / caches / queries JSON translation files.
    LocaleManager -- thread-safe singleton coordinating locale state.

Version: 1.1.0
"""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
SUPPORTED_LOCALES: tuple[str, ...] = ("en", "de", "fr", "es")
DEFAULT_LOCALE: str = "en"
_LOCALE_DIR = Path(__file__).resolve().parent / "locales"

# Files loaded per locale
_CATALOG_FILES: tuple[str, ...] = (
    "esrs_labels.json",
    "report_templates.json",
    "messages.json",
    "glossary.json",
)


# -----------------------------------------------------------------------
# LocaleConfig
# -----------------------------------------------------------------------
@dataclass(frozen=True)
class LocaleConfig:
    """Formatting configuration for a single locale.

    Attributes:
        locale_code: ISO 639-1 language code (e.g. ``"en"``).
        language_name: Human-readable language name in English.
        country_code: Primary ISO 3166-1 alpha-2 country code.
        decimal_separator: Character used as decimal mark.
        thousands_separator: Character used as thousands grouping mark.
        date_format: Short date pattern (Python ``strftime`` style).
        datetime_format: Full datetime pattern.
        long_date_format: Long, human-readable date pattern.
        currency_symbol_position: ``"before"`` or ``"after"`` the amount.
        measurement_system: ``"metric"`` or ``"imperial"``.
        fiscal_year_prefix: Prefix for fiscal-year labels (e.g. ``"FY"``).
    """

    locale_code: str
    language_name: str
    country_code: str
    decimal_separator: str
    thousands_separator: str
    date_format: str
    datetime_format: str
    long_date_format: str
    currency_symbol_position: str  # "before" | "after"
    measurement_system: str  # "metric" | "imperial"
    fiscal_year_prefix: str = "FY"


# Pre-built configs for all supported locales
LOCALE_CONFIGS: Dict[str, LocaleConfig] = {
    "en": LocaleConfig(
        locale_code="en",
        language_name="English",
        country_code="US",
        decimal_separator=".",
        thousands_separator=",",
        date_format="%m/%d/%Y",
        datetime_format="%m/%d/%Y %I:%M %p",
        long_date_format="%B %d, %Y",
        currency_symbol_position="before",
        measurement_system="imperial",
        fiscal_year_prefix="FY",
    ),
    "de": LocaleConfig(
        locale_code="de",
        language_name="German",
        country_code="DE",
        decimal_separator=",",
        thousands_separator=".",
        date_format="%d.%m.%Y",
        datetime_format="%d.%m.%Y %H:%M",
        long_date_format="%d. %B %Y",
        currency_symbol_position="after",
        measurement_system="metric",
        fiscal_year_prefix="GJ",
    ),
    "fr": LocaleConfig(
        locale_code="fr",
        language_name="French",
        country_code="FR",
        decimal_separator=",",
        thousands_separator="\u202f",  # narrow no-break space
        date_format="%d/%m/%Y",
        datetime_format="%d/%m/%Y %H:%M",
        long_date_format="%d %B %Y",
        currency_symbol_position="after",
        measurement_system="metric",
        fiscal_year_prefix="EF",
    ),
    "es": LocaleConfig(
        locale_code="es",
        language_name="Spanish",
        country_code="ES",
        decimal_separator=",",
        thousands_separator=".",
        date_format="%d/%m/%Y",
        datetime_format="%d/%m/%Y %H:%M",
        long_date_format="%d de %B de %Y",
        currency_symbol_position="after",
        measurement_system="metric",
        fiscal_year_prefix="EF",
    ),
}


# -----------------------------------------------------------------------
# TranslationCatalog
# -----------------------------------------------------------------------
class TranslationCatalog:
    """Thread-safe, lazily-loaded translation catalog.

    Loads JSON translation files from disk on first access and caches them
    in memory.  Supports dot-notation key lookup, placeholder interpolation,
    and simple pluralisation (``zero`` / ``one`` / ``many``).

    The fallback chain is:
        1. Requested locale
        2. ``en`` (English)
        3. The raw key string itself

    Example:
        >>> catalog = TranslationCatalog()
        >>> catalog.get("data_points.E1-1.label", "de")
        'Brutto-Scope-1-THG-Emissionen'
    """

    def __init__(self, locales_dir: Optional[Path] = None) -> None:
        """Initialise the catalog.

        Args:
            locales_dir: Override path to the ``locales/`` directory.
                Defaults to ``<package>/locales/``.
        """
        self._locales_dir: Path = locales_dir or _LOCALE_DIR
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._loaded_locales: Set[str] = set()
        self._lock = threading.RLock()
        logger.debug(
            "TranslationCatalog initialised (locales_dir=%s)", self._locales_dir
        )

    # ----- public API --------------------------------------------------

    def get(
        self,
        key: str,
        locale: str = DEFAULT_LOCALE,
        default: Optional[str] = None,
        **params: Any,
    ) -> str:
        """Look up a translation by dot-notation key.

        Args:
            key: Dot-notation key (e.g. ``"data_points.E1-1.label"``).
            locale: Requested locale code.
            default: Explicit default to return when key is missing.  When
                ``None`` the fallback chain is used.
            **params: Interpolation parameters (``str.format_map``).

        Returns:
            Translated string, or fallback.
        """
        self._ensure_loaded(locale)

        # Try requested locale first
        value = self._resolve_key(key, locale)

        # Fallback to English
        if value is None and locale != DEFAULT_LOCALE:
            self._ensure_loaded(DEFAULT_LOCALE)
            value = self._resolve_key(key, DEFAULT_LOCALE)

        # Final fallback
        if value is None:
            value = default if default is not None else key

        # Interpolate parameters
        if params and isinstance(value, str):
            try:
                value = value.format_map(params)
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "Interpolation failed for key=%r locale=%r: %s",
                    key,
                    locale,
                    exc,
                )

        return value

    def get_plural(
        self,
        key: str,
        count: int,
        locale: str = DEFAULT_LOCALE,
        **params: Any,
    ) -> str:
        """Look up a pluralised translation.

        The value at *key* should be a dict with optional ``zero``, ``one``,
        and ``many`` entries.  The fallback is ``many`` -> ``one`` -> raw key.

        Args:
            key: Dot-notation key.
            count: Count determining the plural form.
            locale: Requested locale.
            **params: Interpolation params (``{count}`` is added automatically).

        Returns:
            Pluralised, interpolated string.
        """
        self._ensure_loaded(locale)
        raw = self._resolve_key(key, locale)

        if isinstance(raw, dict):
            if count == 0 and "zero" in raw:
                text = raw["zero"]
            elif count == 1 and "one" in raw:
                text = raw["one"]
            else:
                text = raw.get("many", raw.get("one", key))
        elif isinstance(raw, str):
            text = raw
        else:
            text = key

        params["count"] = count
        try:
            return text.format_map(params)
        except (KeyError, ValueError):
            return text

    def has_key(self, key: str, locale: str = DEFAULT_LOCALE) -> bool:
        """Return ``True`` if *key* exists in *locale*.

        Args:
            key: Dot-notation key.
            locale: Locale code.

        Returns:
            Whether the key exists.
        """
        self._ensure_loaded(locale)
        return self._resolve_key(key, locale) is not None

    def get_all_keys(self, locale: str = DEFAULT_LOCALE) -> List[str]:
        """Return a flat list of all dot-notation keys for *locale*.

        Args:
            locale: Locale code.

        Returns:
            Sorted list of key strings.
        """
        self._ensure_loaded(locale)
        data = self._cache.get(locale, {})
        keys: List[str] = []
        self._flatten_keys(data, "", keys)
        return sorted(keys)

    def reload(self, locale: Optional[str] = None) -> None:
        """Force-reload translation files.

        Args:
            locale: If given, reload only that locale.  Otherwise reload all
                previously loaded locales.
        """
        with self._lock:
            if locale:
                self._loaded_locales.discard(locale)
                self._cache.pop(locale, None)
                self._load_locale(locale)
            else:
                locales = list(self._loaded_locales)
                self._loaded_locales.clear()
                self._cache.clear()
                for loc in locales:
                    self._load_locale(loc)
        logger.info("Translation catalog reloaded (locale=%s)", locale or "all")

    # ----- internal ----------------------------------------------------

    def _ensure_loaded(self, locale: str) -> None:
        """Load locale files if not already cached."""
        if locale not in self._loaded_locales:
            with self._lock:
                if locale not in self._loaded_locales:
                    self._load_locale(locale)

    def _load_locale(self, locale: str) -> None:
        """Read all JSON catalog files for *locale* into the cache."""
        locale_dir = self._locales_dir / locale
        if not locale_dir.is_dir():
            logger.warning("Locale directory does not exist: %s", locale_dir)
            self._loaded_locales.add(locale)
            return

        merged: Dict[str, Any] = {}
        for filename in _CATALOG_FILES:
            filepath = locale_dir / filename
            if not filepath.is_file():
                logger.debug("Catalog file missing: %s", filepath)
                continue
            try:
                with open(filepath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._deep_merge(merged, data)
                logger.debug("Loaded %s (%d top-level keys)", filepath, len(data))
            except (json.JSONDecodeError, OSError) as exc:
                logger.error("Failed to load %s: %s", filepath, exc)

        self._cache[locale] = merged
        self._loaded_locales.add(locale)
        logger.info(
            "Loaded locale %r: %d top-level keys", locale, len(merged)
        )

    def _resolve_key(self, key: str, locale: str) -> Any:
        """Walk dot-notation *key* through the cached dict for *locale*.

        Supports keys that contain hyphens (e.g. ``E1-1``) which are
        common in ESRS data point identifiers.  The split is performed
        only on ``.`` (period).

        Returns:
            The resolved value, or ``None`` if not found.
        """
        data = self._cache.get(locale)
        if data is None:
            return None

        parts = key.split(".")
        current: Any = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    @staticmethod
    def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
        """Recursively merge *overlay* into *base* (mutating *base*)."""
        for key, value in overlay.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                TranslationCatalog._deep_merge(base[key], value)
            else:
                base[key] = value

    @staticmethod
    def _flatten_keys(
        data: Any, prefix: str, result: List[str]
    ) -> None:
        """Recursively collect dot-notation keys from a nested dict."""
        if isinstance(data, dict):
            for k, v in data.items():
                full = f"{prefix}.{k}" if prefix else k
                TranslationCatalog._flatten_keys(v, full, result)
        else:
            if prefix:
                result.append(prefix)


# -----------------------------------------------------------------------
# LocaleManager (thread-safe singleton)
# -----------------------------------------------------------------------
class LocaleManager:
    """Thread-safe locale management singleton.

    Coordinates locale state, translation lookup, and provides convenience
    accessors for ESRS labels, report templates, messages, and glossary
    terms.

    The class implements the Borg pattern so that all instances share the
    same internal state.  This avoids the need for global mutable state
    while still guaranteeing a single logical instance.

    Attributes:
        active_locale: The currently active locale code.

    Example:
        >>> mgr = LocaleManager()
        >>> mgr.set_locale("de")
        >>> mgr.get_esrs_label("E1-1")
        'Brutto-Scope-1-THG-Emissionen'
    """

    _shared_state: Dict[str, Any] = {}
    _init_done: bool = False

    def __init__(
        self,
        locales_dir: Optional[Path] = None,
        default_locale: str = DEFAULT_LOCALE,
    ) -> None:
        """Initialise (or re-attach to) the shared locale manager state.

        Args:
            locales_dir: Override for the locales directory.
            default_locale: Initial active locale code.
        """
        # Borg pattern: share state across instances
        self.__dict__ = self._shared_state

        if not LocaleManager._init_done:
            self._lock = threading.RLock()
            self._catalog = TranslationCatalog(locales_dir=locales_dir)
            self._active_locale: str = default_locale
            self._validate_locale(default_locale)
            LocaleManager._init_done = True
            logger.info(
                "LocaleManager initialised (default=%s, dir=%s)",
                default_locale,
                locales_dir or _LOCALE_DIR,
            )

    # ----- Locale state ------------------------------------------------

    @property
    def active_locale(self) -> str:
        """The currently active locale code."""
        return self._active_locale

    def set_locale(self, locale: str) -> None:
        """Set the active locale.

        Args:
            locale: ISO 639-1 code.

        Raises:
            ValueError: If *locale* is not supported.
        """
        self._validate_locale(locale)
        with self._lock:
            previous = self._active_locale
            self._active_locale = locale
        logger.info("Active locale changed: %s -> %s", previous, locale)

    @contextmanager
    def temporary_locale(self, locale: str) -> Generator[None, None, None]:
        """Context manager for temporary locale switching.

        Restores the previous locale on exit, even if an exception is
        raised inside the block.

        Args:
            locale: Temporary locale to activate.

        Yields:
            Nothing; the temporary locale is active within the block.

        Example:
            >>> mgr = LocaleManager()
            >>> with mgr.temporary_locale("de"):
            ...     mgr.get_message("ui.save")
            'Speichern'
        """
        self._validate_locale(locale)
        with self._lock:
            previous = self._active_locale
            self._active_locale = locale
        try:
            yield
        finally:
            with self._lock:
                self._active_locale = previous
            logger.debug("Restored locale from %s back to %s", locale, previous)

    def get_available_locales(self) -> List[str]:
        """Return the list of supported locale codes.

        Returns:
            Sorted list of locale code strings.
        """
        return sorted(SUPPORTED_LOCALES)

    def get_locale_config(self, locale: Optional[str] = None) -> LocaleConfig:
        """Return the formatting config for *locale*.

        Args:
            locale: Locale code.  Defaults to the active locale.

        Returns:
            A ``LocaleConfig`` instance.

        Raises:
            ValueError: If locale is not supported.
        """
        locale = locale or self._active_locale
        self._validate_locale(locale)
        return LOCALE_CONFIGS[locale]

    # ----- Translation accessors ----------------------------------------

    def get_esrs_label(
        self,
        data_point_id: str,
        locale: Optional[str] = None,
    ) -> str:
        """Return the translated label for an ESRS data point.

        Args:
            data_point_id: Data-point identifier (e.g. ``"E1-1"``).
            locale: Target locale.  Defaults to the active locale.

        Returns:
            Translated label string.
        """
        locale = locale or self._active_locale
        return self._catalog.get(
            f"data_points.{data_point_id}.label", locale
        )

    def get_esrs_description(
        self,
        data_point_id: str,
        locale: Optional[str] = None,
    ) -> str:
        """Return the translated description for an ESRS data point.

        Args:
            data_point_id: Data-point identifier.
            locale: Target locale.

        Returns:
            Translated description string.
        """
        locale = locale or self._active_locale
        return self._catalog.get(
            f"data_points.{data_point_id}.description", locale
        )

    def get_esrs_unit_label(
        self,
        data_point_id: str,
        locale: Optional[str] = None,
    ) -> str:
        """Return the translated unit label for an ESRS data point.

        Args:
            data_point_id: Data-point identifier.
            locale: Target locale.

        Returns:
            Translated unit label.
        """
        locale = locale or self._active_locale
        return self._catalog.get(
            f"data_points.{data_point_id}.unit_label", locale
        )

    def get_standard_name(
        self,
        standard_code: str,
        locale: Optional[str] = None,
    ) -> str:
        """Return the translated name for an ESRS standard.

        Args:
            standard_code: Standard identifier (e.g. ``"E1"``).
            locale: Target locale.

        Returns:
            Translated standard name.
        """
        locale = locale or self._active_locale
        return self._catalog.get(
            f"standards.{standard_code}.name", locale
        )

    def get_standard_description(
        self,
        standard_code: str,
        locale: Optional[str] = None,
    ) -> str:
        """Return the translated description for an ESRS standard.

        Args:
            standard_code: Standard identifier.
            locale: Target locale.

        Returns:
            Translated standard description.
        """
        locale = locale or self._active_locale
        return self._catalog.get(
            f"standards.{standard_code}.description", locale
        )

    def get_report_template(
        self,
        section: str,
        locale: Optional[str] = None,
    ) -> str:
        """Return translated report template text for *section*.

        Args:
            section: Dot-notation path within ``report_templates.json``
                (e.g. ``"cover_page.title"``).
            locale: Target locale.

        Returns:
            Translated template text.
        """
        locale = locale or self._active_locale
        return self._catalog.get(section, locale)

    def get_message(
        self,
        key: str,
        locale: Optional[str] = None,
        **params: Any,
    ) -> str:
        """Return an interpolated translated message.

        This is the most general accessor.  It searches all loaded
        catalog files.

        Args:
            key: Dot-notation key.
            locale: Target locale.
            **params: Interpolation parameters.

        Returns:
            Translated and interpolated string.
        """
        locale = locale or self._active_locale
        return self._catalog.get(key, locale, **params)

    def get_glossary_term(
        self,
        term_key: str,
        locale: Optional[str] = None,
    ) -> Dict[str, str]:
        """Return a glossary entry dict for *term_key*.

        Args:
            term_key: Key within ``terms`` (e.g. ``"double_materiality"``).
            locale: Target locale.

        Returns:
            Dict with ``"term"`` and ``"definition"`` keys, or an empty
            dict if not found.
        """
        locale = locale or self._active_locale
        result = self._catalog.get(f"terms.{term_key}", locale, default=None)
        if isinstance(result, dict):
            return result
        return {"term": term_key, "definition": ""}

    def get_disclosure_requirement(
        self,
        dr_code: str,
        locale: Optional[str] = None,
    ) -> str:
        """Return the translated disclosure requirement name.

        Args:
            dr_code: Disclosure requirement identifier (e.g. ``"E1-1"``).
            locale: Target locale.

        Returns:
            Translated disclosure requirement string.
        """
        locale = locale or self._active_locale
        return self._catalog.get(
            f"disclosure_requirements.{dr_code}", locale
        )

    # ----- Catalog management -------------------------------------------

    @property
    def catalog(self) -> TranslationCatalog:
        """Direct access to the underlying ``TranslationCatalog``."""
        return self._catalog

    def reload(self, locale: Optional[str] = None) -> None:
        """Reload translation files (see ``TranslationCatalog.reload``)."""
        self._catalog.reload(locale)

    # ----- Validation ---------------------------------------------------

    @staticmethod
    def _validate_locale(locale: str) -> None:
        """Raise ``ValueError`` if *locale* is not supported.

        Args:
            locale: Locale code to validate.

        Raises:
            ValueError: If locale is not in ``SUPPORTED_LOCALES``.
        """
        if locale not in SUPPORTED_LOCALES:
            raise ValueError(
                f"Unsupported locale '{locale}'. "
                f"Supported: {', '.join(SUPPORTED_LOCALES)}"
            )

    # ----- Reset (testing) ----------------------------------------------

    @classmethod
    def _reset(cls) -> None:
        """Reset singleton state.  **Intended for tests only.**"""
        cls._shared_state.clear()
        cls._init_done = False
