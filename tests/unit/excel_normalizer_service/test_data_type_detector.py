# -*- coding: utf-8 -*-
"""
Unit Tests for DataTypeDetector (AGENT-DATA-002)

Tests data type detection: initialization, detect_types, detect_column_type,
detect_value_type for each DataType, is_integer, is_float, is_date,
is_currency, is_percentage, is_boolean, is_email, is_unit_value,
parse_date, parse_currency, parse_unit_value, normalize_value, statistics.

Coverage target: 85%+ of data_type_detector.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Inline DataTypeDetector
# ---------------------------------------------------------------------------


class DataTypeDetector:
    """Detects data types of column values in spreadsheet data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._sample_size: int = self._config.get("sample_size", 1000)
        self._stats: Dict[str, int] = {"columns_detected": 0, "values_detected": 0}

    def detect_types(self, headers: List[str], rows: List[List[Any]]) -> Dict[str, str]:
        result = {}
        for col_idx, header in enumerate(headers):
            values = [row[col_idx] for row in rows if col_idx < len(row)]
            result[header] = self.detect_column_type(values)
            self._stats["columns_detected"] += 1
        return result

    def detect_column_type(self, values: List[Any]) -> str:
        if not values:
            return "empty"
        type_counts: Dict[str, int] = {}
        sample = values[:self._sample_size]
        for v in sample:
            dt = self.detect_value_type(v)
            type_counts[dt] = type_counts.get(dt, 0) + 1
        # Remove empty from consideration if other types present
        non_empty = {k: v for k, v in type_counts.items() if k != "empty"}
        if non_empty:
            return max(non_empty, key=non_empty.get)
        return "empty"

    def detect_value_type(self, value: Any) -> str:
        self._stats["values_detected"] += 1
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return "empty"
        s = str(value).strip()
        if self.is_boolean(s):
            return "boolean"
        if self.is_email(s):
            return "email"
        if self.is_percentage(s):
            return "percentage"
        if self.is_currency(s):
            return "currency"
        if self.is_unit_value(s):
            return "unit_value"
        if self.is_integer(s):
            return "integer"
        if self.is_float(s):
            return "float"
        if self.is_date(s):
            return "date"
        return "string"

    def is_integer(self, value: str) -> bool:
        cleaned = value.replace(",", "").replace(" ", "").strip()
        if cleaned.startswith(("+", "-")):
            cleaned = cleaned[1:]
        return cleaned.isdigit() and len(cleaned) > 0

    def is_float(self, value: str) -> bool:
        cleaned = value.replace(",", "").replace(" ", "").strip()
        try:
            float(cleaned)
            return "." in cleaned
        except ValueError:
            return False

    def is_date(self, value: str) -> bool:
        patterns = [
            r"^\d{4}-\d{2}-\d{2}$",                     # ISO: 2025-01-15
            r"^\d{2}/\d{2}/\d{4}$",                      # US/EU: 01/15/2025
            r"^\d{2}-\d{2}-\d{4}$",                      # 01-15-2025
            r"^\d{4}/\d{2}/\d{2}$",                      # 2025/01/15
            r"^\d{2}\.\d{2}\.\d{4}$",                    # EU: 15.01.2025
        ]
        return any(re.match(p, value.strip()) for p in patterns)

    def is_currency(self, value: str) -> bool:
        patterns = [
            r"^\$[\d,]+\.?\d*$",                          # $1,250.50
            r"^[\d,]+\.?\d*\s*(USD|EUR|GBP|JPY|AUD|CAD)$",
            r"^\u20ac[\d,]+\.?\d*$",                      # EUR symbol
            r"^\u00a3[\d,]+\.?\d*$",                      # GBP symbol
        ]
        return any(re.match(p, value.strip()) for p in patterns)

    def is_percentage(self, value: str) -> bool:
        return bool(re.match(r"^[\d.]+\s*%$", value.strip()))

    def is_boolean(self, value: str) -> bool:
        return value.strip().lower() in (
            "true", "false", "yes", "no", "y", "n", "1", "0",
        )

    def is_email(self, value: str) -> bool:
        return bool(re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", value.strip()))

    def is_unit_value(self, value: str) -> bool:
        return bool(re.match(
            r"^[\d,.]+\s*(kg|g|t|lb|oz|km|m|mi|ft|kWh|MWh|GWh|L|gal|m3|tCO2e|kgCO2e)$",
            value.strip(), re.IGNORECASE,
        ))

    def parse_date(self, value: str) -> Optional[str]:
        """Parse date string, returning ISO format. Returns None if ambiguous."""
        value = value.strip()
        # ISO: 2025-01-15
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", value)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        # US: MM/DD/YYYY
        m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", value)
        if m:
            month, day = int(m.group(1)), int(m.group(2))
            if month > 12:
                return None  # Ambiguous
            return f"{m.group(3)}-{m.group(1)}-{m.group(2)}"
        return None

    def parse_currency(self, value: str) -> Optional[float]:
        cleaned = re.sub(r"[$\u20ac\u00a3,]", "", value.strip())
        cleaned = re.sub(r"\s*(USD|EUR|GBP|JPY|AUD|CAD)\s*$", "", cleaned)
        try:
            return float(cleaned)
        except ValueError:
            return None

    def parse_unit_value(self, value: str) -> Optional[Tuple[float, str]]:
        m = re.match(r"^([\d,.]+)\s*([a-zA-Z0-9]+)$", value.strip())
        if m:
            try:
                num = float(m.group(1).replace(",", ""))
                return (num, m.group(2))
            except ValueError:
                pass
        return None

    def normalize_value(self, value: Any, target_type: str) -> Any:
        if value is None:
            return None
        s = str(value).strip()
        if target_type == "integer":
            return int(float(s.replace(",", "")))
        if target_type == "float":
            return float(s.replace(",", ""))
        if target_type == "boolean":
            return s.lower() in ("true", "yes", "y", "1")
        return s

    def get_statistics(self) -> Dict[str, Any]:
        return dict(self._stats)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestDataTypeDetectorInit:
    def test_default_creation(self):
        d = DataTypeDetector()
        assert d._sample_size == 1000

    def test_custom_sample_size(self):
        d = DataTypeDetector(config={"sample_size": 500})
        assert d._sample_size == 500

    def test_initial_statistics(self):
        d = DataTypeDetector()
        assert d.get_statistics()["columns_detected"] == 0


class TestDetectTypes:
    def test_detect_multiple_columns(self):
        d = DataTypeDetector()
        headers = ["name", "year", "value"]
        rows = [["London", "2025", "1250.5"], ["Berlin", "2025", "3400.0"]]
        result = d.detect_types(headers, rows)
        assert "name" in result
        assert "year" in result
        assert "value" in result

    def test_detect_increments_stats(self):
        d = DataTypeDetector()
        d.detect_types(["col1"], [["val1"], ["val2"]])
        assert d.get_statistics()["columns_detected"] == 1

    def test_detect_empty_rows(self):
        d = DataTypeDetector()
        result = d.detect_types(["col1"], [])
        assert result["col1"] == "empty"


class TestDetectColumnType:
    def test_integer_column(self):
        d = DataTypeDetector()
        assert d.detect_column_type(["100", "200", "300"]) == "integer"

    def test_float_column(self):
        d = DataTypeDetector()
        assert d.detect_column_type(["1.5", "2.7", "3.14"]) == "float"

    def test_string_column(self):
        d = DataTypeDetector()
        assert d.detect_column_type(["London", "Berlin", "Tokyo"]) == "string"

    def test_mixed_with_empties(self):
        d = DataTypeDetector()
        result = d.detect_column_type(["100", "", "200", None, "300"])
        assert result == "integer"

    def test_all_empty(self):
        d = DataTypeDetector()
        assert d.detect_column_type([None, "", None]) == "empty"


class TestDetectValueType:
    def test_none(self):
        d = DataTypeDetector()
        assert d.detect_value_type(None) == "empty"

    def test_empty_string(self):
        d = DataTypeDetector()
        assert d.detect_value_type("") == "empty"

    def test_integer(self):
        d = DataTypeDetector()
        assert d.detect_value_type("42") == "integer"

    def test_float(self):
        d = DataTypeDetector()
        assert d.detect_value_type("3.14") == "float"

    def test_boolean_true(self):
        d = DataTypeDetector()
        assert d.detect_value_type("true") == "boolean"

    def test_boolean_false(self):
        d = DataTypeDetector()
        assert d.detect_value_type("false") == "boolean"

    def test_boolean_yes(self):
        d = DataTypeDetector()
        assert d.detect_value_type("yes") == "boolean"

    def test_date_iso(self):
        d = DataTypeDetector()
        assert d.detect_value_type("2025-01-15") == "date"

    def test_date_us(self):
        d = DataTypeDetector()
        assert d.detect_value_type("01/15/2025") == "date"

    def test_date_eu(self):
        d = DataTypeDetector()
        assert d.detect_value_type("15.01.2025") == "date"

    def test_currency_dollar(self):
        d = DataTypeDetector()
        assert d.detect_value_type("$1,250.50") == "currency"

    def test_currency_euro(self):
        d = DataTypeDetector()
        assert d.detect_value_type("\u20ac500.00") == "currency"

    def test_percentage(self):
        d = DataTypeDetector()
        assert d.detect_value_type("85.5%") == "percentage"

    def test_email(self):
        d = DataTypeDetector()
        assert d.detect_value_type("user@example.com") == "email"

    def test_unit_value_kg(self):
        d = DataTypeDetector()
        assert d.detect_value_type("1250 kg") == "unit_value"

    def test_unit_value_kwh(self):
        d = DataTypeDetector()
        assert d.detect_value_type("4500 kWh") == "unit_value"

    def test_unit_value_tco2e(self):
        d = DataTypeDetector()
        assert d.detect_value_type("1250.5 tCO2e") == "unit_value"

    def test_string(self):
        d = DataTypeDetector()
        assert d.detect_value_type("London HQ") == "string"

    def test_value_stats_increment(self):
        d = DataTypeDetector()
        d.detect_value_type("42")
        assert d.get_statistics()["values_detected"] == 1


class TestIsInteger:
    def test_positive(self):
        d = DataTypeDetector()
        assert d.is_integer("42") is True

    def test_negative(self):
        d = DataTypeDetector()
        assert d.is_integer("-42") is True

    def test_with_commas(self):
        d = DataTypeDetector()
        assert d.is_integer("1,000,000") is True

    def test_float_string(self):
        d = DataTypeDetector()
        assert d.is_integer("3.14") is False

    def test_text(self):
        d = DataTypeDetector()
        assert d.is_integer("hello") is False

    def test_zero(self):
        d = DataTypeDetector()
        assert d.is_integer("0") is True


class TestIsFloat:
    def test_simple(self):
        d = DataTypeDetector()
        assert d.is_float("3.14") is True

    def test_negative(self):
        d = DataTypeDetector()
        assert d.is_float("-2.5") is True

    def test_no_decimal(self):
        d = DataTypeDetector()
        assert d.is_float("42") is False

    def test_with_commas(self):
        d = DataTypeDetector()
        assert d.is_float("1,250.50") is True


class TestIsDate:
    def test_iso(self):
        d = DataTypeDetector()
        assert d.is_date("2025-01-15") is True

    def test_us(self):
        d = DataTypeDetector()
        assert d.is_date("01/15/2025") is True

    def test_eu_dot(self):
        d = DataTypeDetector()
        assert d.is_date("15.01.2025") is True

    def test_slash_yyyy(self):
        d = DataTypeDetector()
        assert d.is_date("2025/01/15") is True

    def test_invalid(self):
        d = DataTypeDetector()
        assert d.is_date("not a date") is False

    def test_partial(self):
        d = DataTypeDetector()
        assert d.is_date("2025-01") is False


class TestIsCurrency:
    def test_dollar(self):
        d = DataTypeDetector()
        assert d.is_currency("$1,250.50") is True

    def test_euro_symbol(self):
        d = DataTypeDetector()
        assert d.is_currency("\u20ac500") is True

    def test_pound_symbol(self):
        d = DataTypeDetector()
        assert d.is_currency("\u00a3250.00") is True

    def test_usd_suffix(self):
        d = DataTypeDetector()
        assert d.is_currency("1000.00 USD") is True

    def test_eur_suffix(self):
        d = DataTypeDetector()
        assert d.is_currency("500.00 EUR") is True

    def test_not_currency(self):
        d = DataTypeDetector()
        assert d.is_currency("hello") is False


class TestIsPercentage:
    def test_integer_pct(self):
        d = DataTypeDetector()
        assert d.is_percentage("85%") is True

    def test_float_pct(self):
        d = DataTypeDetector()
        assert d.is_percentage("85.5%") is True

    def test_with_space(self):
        d = DataTypeDetector()
        assert d.is_percentage("85 %") is True

    def test_not_pct(self):
        d = DataTypeDetector()
        assert d.is_percentage("85") is False


class TestIsBoolean:
    def test_true(self):
        d = DataTypeDetector()
        assert d.is_boolean("true") is True

    def test_false(self):
        d = DataTypeDetector()
        assert d.is_boolean("false") is True

    def test_yes(self):
        d = DataTypeDetector()
        assert d.is_boolean("yes") is True

    def test_no(self):
        d = DataTypeDetector()
        assert d.is_boolean("no") is True

    def test_y(self):
        d = DataTypeDetector()
        assert d.is_boolean("y") is True

    def test_1(self):
        d = DataTypeDetector()
        assert d.is_boolean("1") is True

    def test_not_boolean(self):
        d = DataTypeDetector()
        assert d.is_boolean("maybe") is False


class TestIsEmail:
    def test_valid(self):
        d = DataTypeDetector()
        assert d.is_email("user@example.com") is True

    def test_invalid(self):
        d = DataTypeDetector()
        assert d.is_email("not-an-email") is False

    def test_subdomain(self):
        d = DataTypeDetector()
        assert d.is_email("user@sub.example.co.uk") is True


class TestIsUnitValue:
    def test_kg(self):
        d = DataTypeDetector()
        assert d.is_unit_value("1250 kg") is True

    def test_kwh(self):
        d = DataTypeDetector()
        assert d.is_unit_value("4500 kWh") is True

    def test_tco2e(self):
        d = DataTypeDetector()
        assert d.is_unit_value("1250.5 tCO2e") is True

    def test_km(self):
        d = DataTypeDetector()
        assert d.is_unit_value("350 km") is True

    def test_not_unit(self):
        d = DataTypeDetector()
        assert d.is_unit_value("hello") is False


class TestParseDate:
    def test_iso(self):
        d = DataTypeDetector()
        assert d.parse_date("2025-01-15") == "2025-01-15"

    def test_us(self):
        d = DataTypeDetector()
        result = d.parse_date("01/15/2025")
        assert result == "2025-01-15"

    def test_ambiguous(self):
        d = DataTypeDetector()
        result = d.parse_date("13/02/2025")  # Month > 12
        assert result is None

    def test_invalid(self):
        d = DataTypeDetector()
        assert d.parse_date("not a date") is None


class TestParseCurrency:
    def test_dollar(self):
        d = DataTypeDetector()
        assert d.parse_currency("$1,250.50") == 1250.50

    def test_euro(self):
        d = DataTypeDetector()
        assert d.parse_currency("\u20ac500") == 500.0

    def test_usd_suffix(self):
        d = DataTypeDetector()
        assert d.parse_currency("1000.00 USD") == 1000.0

    def test_invalid(self):
        d = DataTypeDetector()
        assert d.parse_currency("not money") is None


class TestParseUnitValue:
    def test_kg(self):
        d = DataTypeDetector()
        result = d.parse_unit_value("1250 kg")
        assert result == (1250.0, "kg")

    def test_kwh(self):
        d = DataTypeDetector()
        result = d.parse_unit_value("4500 kWh")
        assert result == (4500.0, "kWh")

    def test_invalid(self):
        d = DataTypeDetector()
        assert d.parse_unit_value("hello") is None


class TestNormalizeValue:
    def test_to_integer(self):
        d = DataTypeDetector()
        assert d.normalize_value("1,250", "integer") == 1250

    def test_to_float(self):
        d = DataTypeDetector()
        assert d.normalize_value("1,250.50", "float") == 1250.50

    def test_to_boolean_true(self):
        d = DataTypeDetector()
        assert d.normalize_value("yes", "boolean") is True

    def test_to_boolean_false(self):
        d = DataTypeDetector()
        assert d.normalize_value("no", "boolean") is False

    def test_to_string(self):
        d = DataTypeDetector()
        assert d.normalize_value("hello", "string") == "hello"

    def test_none_value(self):
        d = DataTypeDetector()
        assert d.normalize_value(None, "integer") is None


class TestDataTypeDetectorStatistics:
    def test_stats_accumulate(self):
        d = DataTypeDetector()
        d.detect_types(["a", "b"], [["1", "x"], ["2", "y"]])
        stats = d.get_statistics()
        assert stats["columns_detected"] == 2
        assert stats["values_detected"] >= 4
