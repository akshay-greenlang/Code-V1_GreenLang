# -*- coding: utf-8 -*-
"""
Data Type Detector - AGENT-DATA-002: Excel/CSV Normalizer

Automatic data type detection engine that analyses column values to
determine the most likely data type using regex patterns, sampling,
and frequency-based heuristics.

Supports:
    - String, integer, float, decimal detection
    - Date and datetime format recognition (20+ patterns: ISO, US, EU, Asian)
    - Boolean value detection (true/false, yes/no, 1/0, Y/N)
    - Currency detection with symbol recognition (USD, EUR, GBP, JPY, etc.)
    - Percentage detection (50%, 0.5, 50 pct)
    - Unit-value detection (100 kg, 50 kWh, 200 m3)
    - Email pattern detection
    - Sample-based detection for large datasets
    - Date parsing with format detection and ambiguity resolution
    - Currency and unit-value parsing with extraction

Zero-Hallucination Guarantees:
    - All type detections are deterministic regex/pattern-based
    - No LLM calls in detection path
    - Detection confidence is computed from match ratios

Example:
    >>> from greenlang.excel_normalizer.data_type_detector import DataTypeDetector
    >>> detector = DataTypeDetector()
    >>> types = detector.detect_types(rows, headers=["Date", "Amount"])
    >>> print(types)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-002 Excel/CSV Normalizer
Status: Production Ready
"""

from __future__ import annotations

import logging
import re
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "DataType",
    "ColumnTypeResult",
    "DataTypeDetector",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DataType(str, Enum):
    """Detected data type categories."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    EMAIL = "email"
    UNIT_VALUE = "unit_value"
    EMPTY = "empty"
    MIXED = "mixed"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ColumnTypeResult(BaseModel):
    """Type detection result for a single column."""

    column_index: int = Field(default=0, ge=0, description="Zero-based column index")
    column_name: str = Field(default="", description="Column header name")
    detected_type: DataType = Field(
        default=DataType.STRING, description="Primary detected type",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Detection confidence",
    )
    sample_size: int = Field(default=0, ge=0, description="Values sampled")
    non_empty_count: int = Field(default=0, ge=0, description="Non-empty values")
    type_counts: Dict[str, int] = Field(
        default_factory=dict, description="Count per detected type",
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Date format patterns
# ---------------------------------------------------------------------------

DATE_PATTERNS: List[Tuple[str, str]] = [
    # ISO formats
    (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"),
    (r"^\d{4}/\d{2}/\d{2}$", "%Y/%m/%d"),
    (r"^\d{4}\.\d{2}\.\d{2}$", "%Y.%m.%d"),
    # US formats (MM/DD/YYYY)
    (r"^\d{1,2}/\d{1,2}/\d{4}$", "%m/%d/%Y"),
    (r"^\d{1,2}-\d{1,2}-\d{4}$", "%m-%d-%Y"),
    (r"^\d{1,2}\.\d{1,2}\.\d{4}$", "%m.%d.%Y"),
    # EU formats (DD/MM/YYYY) - handled via ambiguity check
    (r"^\d{1,2}/\d{1,2}/\d{2}$", "%m/%d/%y"),
    (r"^\d{1,2}-\d{1,2}-\d{2}$", "%m-%d-%y"),
    # Long formats
    (r"^\w+ \d{1,2},?\s*\d{4}$", "%B %d, %Y"),
    (r"^\d{1,2} \w+ \d{4}$", "%d %B %Y"),
    (r"^\w+ \d{1,2}\s+\d{4}$", "%B %d %Y"),
    (r"^\d{1,2} \w+ \d{4}$", "%d %b %Y"),
    # Short month names
    (r"^\w{3} \d{1,2},?\s*\d{4}$", "%b %d, %Y"),
    (r"^\d{1,2}-\w{3}-\d{4}$", "%d-%b-%Y"),
    (r"^\d{1,2}-\w{3}-\d{2}$", "%d-%b-%y"),
    # ISO datetime
    (r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", "%Y-%m-%dT%H:%M:%S"),
    (r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "%Y-%m-%d %H:%M:%S"),
    (r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$", "%Y-%m-%d %H:%M"),
    # YYYYMMDD compact
    (r"^\d{8}$", "%Y%m%d"),
    # Asian format
    (r"^\d{4}\u5e74\d{1,2}\u6708\d{1,2}\u65e5$", "%Y年%m月%d日"),
]

# Compiled regex for performance
_DATE_REGEXES: List[Tuple[re.Pattern, str]] = [
    (re.compile(pattern), fmt) for pattern, fmt in DATE_PATTERNS
]

# Date parse formats to try in order
_DATE_FORMATS: List[str] = [
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
    "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y",
    "%m.%d.%Y", "%d.%m.%Y",
    "%m/%d/%y", "%d/%m/%y",
    "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
    "%d %B %Y", "%d %b %Y",
    "%d-%b-%Y", "%d-%b-%y",
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
    "%Y%m%d",
]


# ---------------------------------------------------------------------------
# Currency symbols
# ---------------------------------------------------------------------------

CURRENCY_SYMBOLS: Dict[str, str] = {
    "$": "USD", "US$": "USD", "USD": "USD",
    "\u20ac": "EUR", "EUR": "EUR",
    "\u00a3": "GBP", "GBP": "GBP",
    "\u00a5": "JPY", "JPY": "JPY",
    "CNY": "CNY", "RMB": "CNY", "\u5143": "CNY",
    "CHF": "CHF", "Fr.": "CHF",
    "CAD": "CAD", "C$": "CAD",
    "AUD": "AUD", "A$": "AUD",
    "INR": "INR", "\u20b9": "INR",
    "KRW": "KRW", "\u20a9": "KRW",
    "BRL": "BRL", "R$": "BRL",
    "ZAR": "ZAR", "R": "ZAR",
    "SEK": "SEK", "kr": "SEK",
    "NOK": "NOK", "DKK": "DKK",
    "SGD": "SGD", "S$": "SGD",
    "HKD": "HKD", "HK$": "HKD",
    "NZD": "NZD", "NZ$": "NZD",
}

_CURRENCY_PATTERN = re.compile(
    r"^[\s]*"
    r"([A-Z]{2,3}\$?|\$|US\$|C\$|A\$|S\$|HK\$|NZ\$|R\$|"
    r"\u20ac|\u00a3|\u00a5|\u20b9|\u20a9|kr|Fr\.?|R)?"
    r"[\s]*"
    r"(-?[\d,]+\.?\d*)"
    r"[\s]*"
    r"([A-Z]{2,3})?[\s]*$"
)

_PERCENTAGE_PATTERN = re.compile(
    r"^[\s]*(-?[\d,]+\.?\d*)[\s]*(%|pct\.?|percent)[\s]*$",
    re.IGNORECASE,
)

_UNIT_VALUE_PATTERN = re.compile(
    r"^[\s]*(-?[\d,]+\.?\d*)[\s]+"
    r"(kg|g|mg|lb|lbs|oz|t|tonnes?|tons?|"
    r"kWh|MWh|GWh|Wh|kW|MW|GW|W|"
    r"m3|m2|m|km|cm|mm|ft|sqft|sq\.?\s*ft|"
    r"L|liters?|litres?|gal|gallons?|"
    r"tCO2e?|CO2e?|"
    r"therms?|BTU|MMBtu|GJ|MJ|kJ|"
    r"ppm|ppb)[\s]*$",
    re.IGNORECASE,
)

_EMAIL_PATTERN = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)

_BOOLEAN_VALUES = {
    "true", "false", "yes", "no", "y", "n",
    "1", "0", "t", "f", "on", "off",
    "ja", "nein", "oui", "non", "si",
}


# ---------------------------------------------------------------------------
# DataTypeDetector
# ---------------------------------------------------------------------------


class DataTypeDetector:
    """Automatic data type detection engine.

    Analyses column values to determine the most likely data type using
    regex patterns, sampling, and frequency-based voting. Each column
    receives a primary type and confidence score based on the fraction
    of values matching that type.

    Attributes:
        _config: Configuration dictionary.
        _lock: Threading lock for statistics.
        _stats: Detection statistics.

    Example:
        >>> detector = DataTypeDetector()
        >>> types = detector.detect_types([["2024-01-01", "100"]], headers=["Date", "Value"])
        >>> print(types[0].detected_type)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise DataTypeDetector.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``sample_size``: int (default 1000)
                - ``confidence_threshold``: float (default 0.6)
        """
        self._config = config or {}
        self._sample_size: int = self._config.get("sample_size", 1000)
        self._confidence_threshold: float = self._config.get(
            "confidence_threshold", 0.6,
        )
        self._lock = threading.Lock()
        self._stats: Dict[str, int] = {
            "values_detected": 0,
            "type_string": 0,
            "type_integer": 0,
            "type_float": 0,
            "type_boolean": 0,
            "type_date": 0,
            "type_datetime": 0,
            "type_currency": 0,
            "type_percentage": 0,
            "type_email": 0,
            "type_unit_value": 0,
            "type_empty": 0,
            "parse_successes": 0,
            "parse_failures": 0,
        }
        logger.info(
            "DataTypeDetector initialised: sample_size=%d",
            self._sample_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_types(
        self,
        rows: List[List[Any]],
        headers: Optional[List[str]] = None,
        sample_size: Optional[int] = None,
    ) -> List[ColumnTypeResult]:
        """Detect data types for all columns in a dataset.

        Args:
            rows: List of data rows.
            headers: Optional column header names.
            sample_size: Override sample size (default uses instance config).

        Returns:
            List of ColumnTypeResult objects (one per column).
        """
        start = time.monotonic()

        if not rows:
            return []

        effective_sample = sample_size or self._sample_size
        sample_rows = rows[:effective_sample]

        # Determine column count
        col_count = max((len(r) for r in sample_rows), default=0)
        if headers and len(headers) > col_count:
            col_count = len(headers)

        results: List[ColumnTypeResult] = []
        for col_idx in range(col_count):
            # Collect column values
            values = []
            for row in sample_rows:
                if col_idx < len(row):
                    values.append(row[col_idx])
                else:
                    values.append(None)

            col_name = headers[col_idx] if headers and col_idx < len(headers) else f"column_{col_idx}"
            result = self.detect_column_type(values)
            result.column_index = col_idx
            result.column_name = col_name
            results.append(result)

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Detected types for %d columns (%d sample rows, %.1f ms)",
            col_count, len(sample_rows), elapsed,
        )
        return results

    def detect_column_type(self, values: List[Any]) -> ColumnTypeResult:
        """Detect the dominant data type for a single column.

        Samples values, detects the type of each, and votes to determine
        the primary type.

        Args:
            values: List of cell values for the column.

        Returns:
            ColumnTypeResult with the detected type and confidence.
        """
        type_counts: Dict[str, int] = {}
        non_empty = 0
        total = len(values)

        for value in values:
            detected = self.detect_value_type(value)
            type_name = detected.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            if detected != DataType.EMPTY:
                non_empty += 1

        with self._lock:
            self._stats["values_detected"] += total

        # Vote for primary type (exclude EMPTY from voting)
        non_empty_counts = {
            k: v for k, v in type_counts.items() if k != DataType.EMPTY.value
        }

        if not non_empty_counts:
            return ColumnTypeResult(
                detected_type=DataType.EMPTY,
                confidence=1.0,
                sample_size=total,
                non_empty_count=0,
                type_counts=type_counts,
            )

        # Find most common non-empty type
        primary_type_name = max(non_empty_counts, key=non_empty_counts.get)  # type: ignore[arg-type]
        primary_count = non_empty_counts[primary_type_name]
        confidence = primary_count / max(non_empty, 1)

        # Check for mixed type
        if confidence < self._confidence_threshold and len(non_empty_counts) > 1:
            detected = DataType.MIXED
        else:
            detected = DataType(primary_type_name)

        # Update stats
        stat_key = f"type_{detected.value}"
        with self._lock:
            if stat_key in self._stats:
                self._stats[stat_key] += 1

        return ColumnTypeResult(
            detected_type=detected,
            confidence=round(confidence, 4),
            sample_size=total,
            non_empty_count=non_empty,
            type_counts=type_counts,
        )

    def detect_value_type(self, value: Any) -> DataType:
        """Detect the data type of a single value.

        Args:
            value: Cell value (string, number, None, etc.).

        Returns:
            Detected DataType.
        """
        # Handle None and empty
        if value is None:
            return DataType.EMPTY

        # Handle native Python types
        if isinstance(value, bool):
            return DataType.BOOLEAN
        if isinstance(value, int):
            return DataType.INTEGER
        if isinstance(value, float):
            return DataType.FLOAT
        if isinstance(value, datetime):
            return DataType.DATETIME

        # Convert to string for pattern matching
        s = str(value).strip()
        if not s:
            return DataType.EMPTY

        # Check patterns in specificity order
        if self.is_boolean(s):
            return DataType.BOOLEAN
        if self.is_email(s):
            return DataType.EMAIL
        if self.is_percentage(s):
            return DataType.PERCENTAGE
        if self.is_currency(s):
            return DataType.CURRENCY
        if self.is_unit_value(s):
            return DataType.UNIT_VALUE
        if self.is_date(s):
            # Distinguish date vs datetime
            if "T" in s or (":" in s and re.search(r"\d{2}:\d{2}", s)):
                return DataType.DATETIME
            return DataType.DATE
        if self.is_integer(s):
            return DataType.INTEGER
        if self.is_float(s):
            return DataType.FLOAT

        return DataType.STRING

    def is_integer(self, value: str) -> bool:
        """Check if a string represents an integer.

        Args:
            value: String value.

        Returns:
            True if the value is an integer.
        """
        cleaned = value.strip().replace(",", "").replace(" ", "")
        if not cleaned:
            return False
        # Allow optional leading sign
        if cleaned.startswith(("+", "-")):
            cleaned = cleaned[1:]
        return cleaned.isdigit()

    def is_float(self, value: str) -> bool:
        """Check if a string represents a float/decimal number.

        Args:
            value: String value.

        Returns:
            True if the value is a float.
        """
        cleaned = value.strip().replace(",", "").replace(" ", "")
        if not cleaned:
            return False
        try:
            float(cleaned)
            return "." in cleaned or "e" in cleaned.lower()
        except ValueError:
            return False

    def is_date(self, value: str) -> bool:
        """Check if a string matches any known date format pattern.

        Args:
            value: String value.

        Returns:
            True if the value looks like a date.
        """
        cleaned = value.strip()
        if not cleaned or len(cleaned) < 6:
            return False

        for regex, fmt in _DATE_REGEXES:
            if regex.match(cleaned):
                return True

        # Try parsing as fallback
        for fmt in _DATE_FORMATS:
            try:
                datetime.strptime(cleaned, fmt)
                return True
            except ValueError:
                continue

        return False

    def is_currency(self, value: str) -> bool:
        """Check if a string represents a currency value.

        Args:
            value: String value.

        Returns:
            True if the value contains a currency symbol/code and a number.
        """
        return _CURRENCY_PATTERN.match(value.strip()) is not None and bool(
            _CURRENCY_PATTERN.match(value.strip()).group(1) or  # type: ignore[union-attr]
            _CURRENCY_PATTERN.match(value.strip()).group(3)  # type: ignore[union-attr]
        )

    def is_percentage(self, value: str) -> bool:
        """Check if a string represents a percentage value.

        Args:
            value: String value.

        Returns:
            True if the value contains a % sign or 'pct' suffix.
        """
        return _PERCENTAGE_PATTERN.match(value.strip()) is not None

    def is_boolean(self, value: str) -> bool:
        """Check if a string represents a boolean value.

        Args:
            value: String value.

        Returns:
            True if the value is a known boolean representation.
        """
        return value.strip().lower() in _BOOLEAN_VALUES

    def is_email(self, value: str) -> bool:
        """Check if a string represents an email address.

        Args:
            value: String value.

        Returns:
            True if the value matches the email pattern.
        """
        return _EMAIL_PATTERN.match(value.strip()) is not None

    def is_unit_value(self, value: str) -> bool:
        """Check if a string represents a value with a unit.

        Args:
            value: String value (e.g. "100 kg", "50 kWh").

        Returns:
            True if the value matches the unit-value pattern.
        """
        return _UNIT_VALUE_PATTERN.match(value.strip()) is not None

    def parse_date(self, value: str) -> Optional[datetime]:
        """Parse a date string with format detection.

        Tries each known format in order. For ambiguous dates (where
        day and month could be swapped), prefers the ISO/US interpretation
        unless the day value exceeds 12.

        Args:
            value: Date string to parse.

        Returns:
            Parsed datetime or None if unparseable.
        """
        cleaned = value.strip()
        if not cleaned:
            return None

        for fmt in _DATE_FORMATS:
            try:
                dt = datetime.strptime(cleaned, fmt)
                with self._lock:
                    self._stats["parse_successes"] += 1
                return dt
            except ValueError:
                continue

        with self._lock:
            self._stats["parse_failures"] += 1
        return None

    def parse_currency(self, value: str) -> Optional[Tuple[float, str]]:
        """Parse a currency string and extract amount and currency code.

        Args:
            value: Currency string (e.g. "$1,234.56", "EUR 100").

        Returns:
            Tuple of (amount, currency_code) or None.
        """
        match = _CURRENCY_PATTERN.match(value.strip())
        if not match:
            return None

        prefix_symbol = match.group(1) or ""
        amount_str = match.group(2).replace(",", "")
        suffix_code = match.group(3) or ""

        try:
            amount = float(amount_str)
        except ValueError:
            return None

        # Resolve currency code
        symbol = prefix_symbol.strip() or suffix_code.strip()
        currency_code = CURRENCY_SYMBOLS.get(symbol, symbol.upper() if symbol else "USD")

        with self._lock:
            self._stats["parse_successes"] += 1
        return amount, currency_code

    def parse_unit_value(self, value: str) -> Optional[Tuple[float, str]]:
        """Parse a value+unit string and extract numeric value and unit.

        Args:
            value: Unit-value string (e.g. "100 kg", "50.5 kWh").

        Returns:
            Tuple of (numeric_value, unit_string) or None.
        """
        match = _UNIT_VALUE_PATTERN.match(value.strip())
        if not match:
            return None

        amount_str = match.group(1).replace(",", "")
        unit = match.group(2).strip()

        try:
            amount = float(amount_str)
        except ValueError:
            return None

        with self._lock:
            self._stats["parse_successes"] += 1
        return amount, unit

    def normalize_value(self, value: Any, target_type: DataType) -> Any:
        """Convert a value to a target data type.

        Args:
            value: Value to convert.
            target_type: Target DataType.

        Returns:
            Converted value, or original value if conversion fails.
        """
        if value is None:
            return None

        s = str(value).strip()
        if not s:
            return None

        try:
            if target_type == DataType.INTEGER:
                return int(float(s.replace(",", "")))
            elif target_type == DataType.FLOAT:
                return float(s.replace(",", ""))
            elif target_type == DataType.BOOLEAN:
                return s.lower() in {"true", "yes", "y", "1", "t", "on", "ja", "oui", "si"}
            elif target_type == DataType.DATE:
                dt = self.parse_date(s)
                return dt.strftime("%Y-%m-%d") if dt else s
            elif target_type == DataType.DATETIME:
                dt = self.parse_date(s)
                return dt.isoformat() if dt else s
            elif target_type == DataType.CURRENCY:
                result = self.parse_currency(s)
                return result[0] if result else s
            elif target_type == DataType.PERCENTAGE:
                match = _PERCENTAGE_PATTERN.match(s)
                if match:
                    return float(match.group(1).replace(",", ""))
                return s
            elif target_type == DataType.UNIT_VALUE:
                result = self.parse_unit_value(s)
                return result[0] if result else s
            else:
                return s
        except (ValueError, TypeError, AttributeError):
            logger.debug("Failed to normalize '%s' to %s", s, target_type.value)
            return value

    def get_statistics(self) -> Dict[str, Any]:
        """Return detection statistics.

        Returns:
            Dictionary with counter values and detection breakdown.
        """
        with self._lock:
            return {
                "values_detected": self._stats["values_detected"],
                "type_counts": {
                    k.replace("type_", ""): v
                    for k, v in self._stats.items()
                    if k.startswith("type_")
                },
                "parse_successes": self._stats["parse_successes"],
                "parse_failures": self._stats["parse_failures"],
                "timestamp": _utcnow().isoformat(),
            }
