# -*- coding: utf-8 -*-
"""
Unit Tests for ValidityChecker Engine - AGENT-DATA-010 (GL-DATA-X-013)
======================================================================

Tests ValidityChecker from greenlang.data_quality_profiler.validity_checker.

Covers:
    - Initialization (default/custom config, stats, None config)
    - Full dataset validation (return type, per-column, score, edge cases)
    - Per-column validation (conformance, formats, ranges, regex, domain)
    - Type conformance checking (string, int, float, boolean, date, etc.)
    - Format checking (30+ parametrized tests for 20+ format types)
    - Range checking (in range, below min, above max, boundaries)
    - Regex checking (matching, non-matching, complex, case sensitivity)
    - Domain checking (in set, not in set, case sensitivity)
    - Cross-field validation (less than, equals, not equals, missing fields)
    - Validity score computation
    - Issue generation
    - Aggregate statistics
    - Provenance hashing
    - Thread safety

Target: 120+ tests, ~1050 lines.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List

import pytest

from greenlang.data_quality_profiler.validity_checker import (
    ValidityChecker,
    _luhn_check,
    _is_integer,
    _is_float,
    _is_boolean,
    _is_string,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
    SEVERITY_INFO,
    OPERATOR_EQUALS,
    OPERATOR_NOT_EQUALS,
    OPERATOR_GREATER_THAN,
    OPERATOR_LESS_THAN,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def checker():
    """Create a ValidityChecker with default config."""
    return ValidityChecker()


@pytest.fixture
def strict_checker():
    """Create a ValidityChecker with strict mode."""
    return ValidityChecker(config={"strict_mode": True, "max_issues": 100})


@pytest.fixture
def custom_format_checker():
    """Create a ValidityChecker with a custom format pattern."""
    return ValidityChecker(config={
        "custom_formats": {
            "product_code": r"^PRD-\d{6}$",
        }
    })


@pytest.fixture
def valid_email_data():
    """Dataset with all valid emails."""
    return [
        {"email": "alice@example.com", "age": 30},
        {"email": "bob@test.org", "age": 25},
        {"email": "charlie@corp.net", "age": 35},
    ]


@pytest.fixture
def mixed_email_data():
    """Dataset with mix of valid and invalid emails."""
    return [
        {"email": "alice@example.com", "age": 30},
        {"email": "invalid-email", "age": 25},
        {"email": "bob@test.org", "age": 35},
        {"email": "no-domain@", "age": 28},
    ]


# ============================================================================
# TestInit
# ============================================================================


class TestInit:
    """Test ValidityChecker initialization."""

    def test_default_config(self):
        """Test default configuration values."""
        checker = ValidityChecker()
        assert checker._strict_mode is False
        assert checker._max_issues == 1000

    def test_custom_config(self):
        """Test custom configuration overrides."""
        checker = ValidityChecker(config={
            "strict_mode": True,
            "max_issues": 500,
        })
        assert checker._strict_mode is True
        assert checker._max_issues == 500

    def test_initial_stats(self):
        """Test statistics are zeroed on init."""
        checker = ValidityChecker()
        stats = checker.get_statistics()
        assert stats["validations_completed"] == 0
        assert stats["total_violations"] == 0

    def test_none_config(self):
        """Test None config uses defaults."""
        checker = ValidityChecker(config=None)
        assert checker._strict_mode is False


# ============================================================================
# TestValidate
# ============================================================================


class TestValidate:
    """Test full dataset validation."""

    def test_returns_dict(self, checker, valid_email_data):
        """Test validate() returns a dictionary."""
        rules = [{"column": "email", "format_type": "email"}]
        result = checker.validate(valid_email_data, rules=rules)
        assert isinstance(result, dict)

    def test_per_column_validity(self, checker, mixed_email_data):
        """Test column_validity contains results for validated columns."""
        rules = [{"column": "email", "format_type": "email"}]
        result = checker.validate(mixed_email_data, rules=rules)
        assert "email" in result["column_validity"]

    def test_overall_score(self, checker, valid_email_data):
        """Test overall validity_score for all valid data."""
        rules = [{"column": "email", "format_type": "email"}]
        result = checker.validate(valid_email_data, rules=rules)
        assert result["validity_score"] == pytest.approx(1.0, abs=0.01)

    def test_empty_data_raises_error(self, checker):
        """Test empty data raises ValueError."""
        with pytest.raises(ValueError, match="Cannot validate empty dataset"):
            checker.validate([])

    def test_all_valid(self, checker, valid_email_data):
        """Test all valid data produces score 1.0."""
        rules = [{"column": "email", "format_type": "email"}]
        result = checker.validate(valid_email_data, rules=rules)
        assert result["validity_score"] == pytest.approx(1.0, abs=0.01)

    def test_mixed_types(self, checker):
        """Test mixed type data produces violations."""
        data = [{"val": 1}, {"val": "not_int"}, {"val": 3}]
        rules = [{"column": "val", "expected_type": "integer"}]
        result = checker.validate(data, rules=rules)
        assert result["violation_count"] > 0

    def test_format_issues(self, checker, mixed_email_data):
        """Test format violations detected."""
        rules = [{"column": "email", "format_type": "email"}]
        result = checker.validate(mixed_email_data, rules=rules)
        assert result["violation_count"] > 0

    def test_range_issues(self, checker):
        """Test range violations detected."""
        data = [{"age": 25}, {"age": -5}, {"age": 200}]
        rules = [{"column": "age", "min_val": 0, "max_val": 150}]
        result = checker.validate(data, rules=rules)
        assert result["violation_count"] > 0

    def test_cross_field_issues(self, checker):
        """Test cross-field violations detected."""
        data = [
            {"start": 10, "end": 20},
            {"start": 30, "end": 20},  # start > end
        ]
        rules = [{
            "column": "start",
            "cross_field": {
                "field_a": "start",
                "operator": "LESS_THAN",
                "field_b": "end",
            },
        }]
        result = checker.validate(data, rules=rules)
        assert len(result["cross_field_violations"]) > 0

    def test_provenance_hash(self, checker, valid_email_data):
        """Test provenance_hash is 64-char hex string."""
        rules = [{"column": "email", "format_type": "email"}]
        result = checker.validate(valid_email_data, rules=rules)
        assert len(result["provenance_hash"]) == 64

    def test_issues_list(self, checker, mixed_email_data):
        """Test issues list is returned."""
        rules = [{"column": "email", "format_type": "email"}]
        result = checker.validate(mixed_email_data, rules=rules)
        assert "issues" in result
        assert isinstance(result["issues"], list)

    def test_no_rules_score_1(self, checker, mixed_email_data):
        """Test no rules produces validity score 1.0."""
        result = checker.validate(mixed_email_data, rules=[])
        assert result["validity_score"] == pytest.approx(1.0, abs=0.01)


# ============================================================================
# TestValidateColumn
# ============================================================================


class TestValidateColumn:
    """Test per-column validation."""

    def test_all_conforming(self, checker):
        """Test all conforming values produce validity_rate 1.0."""
        result = checker.validate_column(
            [1, 2, 3, 4, 5], "num_col", expected_type="integer"
        )
        assert result["validity_rate"] == pytest.approx(1.0, abs=0.01)

    def test_mixed_types(self, checker):
        """Test mixed types produce violations."""
        result = checker.validate_column(
            [1, "not_int", 3], "col", expected_type="integer"
        )
        assert result["invalid_count"] > 0

    def test_expected_type_override(self, checker):
        """Test expected_type overrides auto-detection."""
        result = checker.validate_column(
            ["hello", "world"], "col", expected_type="integer"
        )
        assert result["invalid_count"] == 2

    def test_format_pattern(self, checker):
        """Test format_pattern validation."""
        result = checker.validate_column(
            ["alice@example.com", "invalid", "bob@test.org"],
            "email_col",
            format_pattern="email",
        )
        assert result["invalid_count"] == 1

    def test_no_non_null_values(self, checker):
        """Test null values are not treated as violations."""
        result = checker.validate_column(
            [None, None, None], "col", expected_type="integer"
        )
        assert result["valid_count"] == 3
        assert result["invalid_count"] == 0

    def test_single_value(self, checker):
        """Test single value validation."""
        result = checker.validate_column(
            ["alice@example.com"], "col", format_pattern="email"
        )
        assert result["validity_rate"] == pytest.approx(1.0, abs=0.01)

    def test_column_name_in_result(self, checker):
        """Test column_name is preserved."""
        result = checker.validate_column([1, 2, 3], "my_col")
        assert result["column_name"] == "my_col"

    def test_with_range(self, checker):
        """Test range validation."""
        result = checker.validate_column(
            [5, 10, 15, 20, 25], "col", min_val=0, max_val=20,
        )
        assert result["invalid_count"] == 1  # 25 > 20

    def test_with_regex(self, checker):
        """Test regex validation."""
        result = checker.validate_column(
            ["ABC-123", "DEF-456", "invalid"], "col",
            regex_pattern=r"^[A-Z]{3}-\d{3}$",
        )
        assert result["invalid_count"] == 1

    def test_with_allowed_values(self, checker):
        """Test domain validation."""
        result = checker.validate_column(
            ["red", "green", "blue", "purple"], "col",
            allowed_values=["red", "green", "blue"],
        )
        assert result["invalid_count"] == 1


# ============================================================================
# TestCheckTypeConformance
# ============================================================================


class TestCheckTypeConformance:
    """Test type conformance checking."""

    def test_all_string(self, checker):
        """Test all string conformance = 1.0."""
        result = checker.check_type_conformance(
            ["hello", "world", "test"], "string"
        )
        assert result == pytest.approx(1.0, abs=0.01)

    def test_all_int(self, checker):
        """Test all int conformance = 1.0."""
        result = checker.check_type_conformance([1, 2, 3], "integer")
        assert result == pytest.approx(1.0, abs=0.01)

    def test_mixed_string_int(self, checker):
        """Test mixed string/int conformance < 1.0."""
        result = checker.check_type_conformance([1, "hello", 3], "integer")
        assert result < 1.0

    def test_all_float(self, checker):
        """Test all float conformance = 1.0."""
        result = checker.check_type_conformance([1.5, 2.7, 3.14], "float")
        assert result == pytest.approx(1.0, abs=0.01)

    def test_boolean_values(self, checker):
        """Test boolean conformance."""
        result = checker.check_type_conformance([True, False, True], "boolean")
        assert result == pytest.approx(1.0, abs=0.01)

    def test_date_values(self, checker):
        """Test date conformance."""
        result = checker.check_type_conformance(
            ["2025-01-15", "2025-06-30"], "date"
        )
        assert result == pytest.approx(1.0, abs=0.01)

    def test_empty_list(self, checker):
        """Test empty list returns 1.0."""
        result = checker.check_type_conformance([], "integer")
        assert result == pytest.approx(1.0, abs=0.01)

    def test_single_value(self, checker):
        """Test single value conformance."""
        result = checker.check_type_conformance([42], "integer")
        assert result == pytest.approx(1.0, abs=0.01)

    def test_unknown_type(self, checker):
        """Test unknown type returns 1.0 (permissive)."""
        result = checker.check_type_conformance([1, 2], "unknown_type_xyz")
        assert result == pytest.approx(1.0, abs=0.01)

    def test_null_values_excluded(self, checker):
        """Test None values excluded from conformance check."""
        result = checker.check_type_conformance([1, None, 3], "integer")
        assert result == pytest.approx(1.0, abs=0.01)


# ============================================================================
# TestCheckFormat (parametrized for 30+ format checks)
# ============================================================================


class TestCheckFormat:
    """Test format validation with parametrized tests for 20+ formats."""

    # --- Email ---
    @pytest.mark.parametrize("value,expected", [
        ("alice@example.com", True),
        ("bob+tag@test.org", True),
        ("user.name@corp.net", True),
        ("invalid-email", False),
        ("@no-local.com", False),
        ("no-domain@", False),
    ])
    def test_email_format(self, checker, value, expected):
        """Test email format validation."""
        assert checker.check_format(value, "email") is expected

    # --- Phone ---
    @pytest.mark.parametrize("value,expected", [
        ("+1-555-0101", True),
        ("+44 20 7946 0958", True),
        ("+49-30-901820", True),
        ("abc", False),
        ("12", False),
        ("", False),
    ])
    def test_phone_format(self, checker, value, expected):
        """Test phone format validation."""
        assert checker.check_format(value, "phone") is expected

    # --- URL ---
    @pytest.mark.parametrize("value,expected", [
        ("https://example.com", True),
        ("http://test.org/path", True),
        ("https://api.corp.net/v1/data", True),
        ("ftp://invalid.com", False),
        ("just-text", False),
        ("", False),
    ])
    def test_url_format(self, checker, value, expected):
        """Test URL format validation."""
        assert checker.check_format(value, "url") is expected

    # --- IPv4 ---
    @pytest.mark.parametrize("value,expected", [
        ("192.168.1.1", True),
        ("10.0.0.1", True),
        ("999.999.999.999", False),
        ("not-an-ip", False),
    ])
    def test_ipv4_format(self, checker, value, expected):
        """Test IPv4 format validation."""
        assert checker.check_format(value, "ipv4") is expected

    # --- IPv6 ---
    @pytest.mark.parametrize("value,expected", [
        ("2001:0db8:85a3:0000:0000:8a2e:0370:7334", True),
    ])
    def test_ipv6_format(self, checker, value, expected):
        """Test IPv6 format validation."""
        assert checker.check_format(value, "ipv6") is expected

    # --- UUID ---
    @pytest.mark.parametrize("value,expected", [
        ("550e8400-e29b-41d4-a716-446655440000", True),
        ("not-a-uuid", False),
        ("550e8400e29b41d4a716446655440000", False),
    ])
    def test_uuid_format(self, checker, value, expected):
        """Test UUID format validation."""
        assert checker.check_format(value, "uuid") is expected

    # --- Date ISO ---
    @pytest.mark.parametrize("value,expected", [
        ("2025-01-15", True),
        ("2025-12-31", True),
        ("2025-13-01", False),
        ("not-a-date", False),
    ])
    def test_date_iso_format(self, checker, value, expected):
        """Test ISO date format validation."""
        assert checker.check_format(value, "date_iso") is expected

    # --- Date US ---
    def test_date_us_format(self, checker):
        """Test US date format validation."""
        assert checker.check_format("01/15/2025", "date_us") is True

    # --- Date EU ---
    def test_date_eu_format(self, checker):
        """Test EU date format validation."""
        assert checker.check_format("15/01/2025", "date_eu") is True

    # --- Datetime ISO ---
    def test_datetime_iso_format(self, checker):
        """Test ISO datetime format validation."""
        assert checker.check_format("2025-01-15T10:30:00Z", "datetime_iso") is True
        assert checker.check_format("not-datetime", "datetime_iso") is False

    # --- Currency ---
    def test_currency_format(self, checker):
        """Test currency format validation."""
        assert checker.check_format("$100.00", "currency") is True
        assert checker.check_format("invalid", "currency") is False

    # --- Percentage ---
    def test_percentage_format(self, checker):
        """Test percentage format validation."""
        assert checker.check_format("50%", "percentage") is True
        assert checker.check_format("abc%", "percentage") is False

    # --- ZIP US ---
    def test_zip_us_format(self, checker):
        """Test US ZIP code format validation."""
        assert checker.check_format("12345", "zip_code_us") is True
        assert checker.check_format("12345-6789", "zip_code_us") is True
        assert checker.check_format("123", "zip_code_us") is False

    # --- ZIP UK ---
    def test_zip_uk_format(self, checker):
        """Test UK postal code format validation."""
        assert checker.check_format("SW1A 1AA", "zip_code_uk") is True
        assert checker.check_format("12345", "zip_code_uk") is False

    # --- Country ISO2 ---
    def test_country_iso2_format(self, checker):
        """Test ISO 3166-1 alpha-2 country code format."""
        assert checker.check_format("US", "country_code_iso2") is True
        assert checker.check_format("USA", "country_code_iso2") is False

    # --- Country ISO3 ---
    def test_country_iso3_format(self, checker):
        """Test ISO 3166-1 alpha-3 country code format."""
        assert checker.check_format("USA", "country_code_iso3") is True
        assert checker.check_format("US", "country_code_iso3") is False

    # --- Latitude ---
    def test_latitude_format(self, checker):
        """Test latitude format validation."""
        assert checker.check_format("45.1234", "latitude") is True
        assert checker.check_format("-90.0", "latitude") is True
        assert checker.check_format("91.0", "latitude") is False

    # --- Longitude ---
    def test_longitude_format(self, checker):
        """Test longitude format validation."""
        assert checker.check_format("120.5678", "longitude") is True
        assert checker.check_format("-180.0", "longitude") is True
        assert checker.check_format("181.0", "longitude") is False

    # --- Hex Color ---
    def test_hex_color_format(self, checker):
        """Test hex color format validation."""
        assert checker.check_format("#FF0000", "hex_color") is True
        assert checker.check_format("#abc", "hex_color") is True
        assert checker.check_format("red", "hex_color") is False

    # --- Unknown format ---
    def test_unknown_format(self, checker):
        """Test unknown format type returns True (permissive)."""
        assert checker.check_format("anything", "unknown_format_xyz") is True

    # --- None value ---
    def test_none_value(self, checker):
        """Test None value returns False for any format."""
        assert checker.check_format(None, "email") is False

    # --- Custom format ---
    def test_custom_format(self, custom_format_checker):
        """Test custom format pattern validation."""
        assert custom_format_checker.check_format("PRD-123456", "product_code") is True
        assert custom_format_checker.check_format("PRD-12", "product_code") is False


# ============================================================================
# TestCheckRange
# ============================================================================


class TestCheckRange:
    """Test range validation."""

    def test_in_range(self, checker):
        """Test value within range passes."""
        assert checker.check_range(50, min_val=0, max_val=100) is True

    def test_below_min(self, checker):
        """Test value below min fails."""
        assert checker.check_range(-5, min_val=0, max_val=100) is False

    def test_above_max(self, checker):
        """Test value above max fails."""
        assert checker.check_range(150, min_val=0, max_val=100) is False

    def test_equal_to_min(self, checker):
        """Test value equal to min passes (inclusive)."""
        assert checker.check_range(0, min_val=0, max_val=100) is True

    def test_equal_to_max(self, checker):
        """Test value equal to max passes (inclusive)."""
        assert checker.check_range(100, min_val=0, max_val=100) is True

    def test_no_min(self, checker):
        """Test no min constraint."""
        assert checker.check_range(-999, max_val=100) is True

    def test_no_max(self, checker):
        """Test no max constraint."""
        assert checker.check_range(999, min_val=0) is True

    def test_string_comparison(self, checker):
        """Test string values converted to numeric for range check."""
        assert checker.check_range("50", min_val=0, max_val=100) is True
        assert checker.check_range("not_a_number", min_val=0, max_val=100) is False


# ============================================================================
# TestCheckRegex
# ============================================================================


class TestCheckRegex:
    """Test regex validation."""

    def test_matching(self, checker):
        """Test matching regex passes."""
        assert checker.check_regex("ABC-123", r"^[A-Z]{3}-\d{3}$") is True

    def test_non_matching(self, checker):
        """Test non-matching regex fails."""
        assert checker.check_regex("abc-123", r"^[A-Z]{3}-\d{3}$") is False

    def test_complex_pattern(self, checker):
        """Test complex regex pattern."""
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"
        assert checker.check_regex("2025-01-15T10:30:00", pattern) is True

    def test_case_sensitivity(self, checker):
        """Test regex is case sensitive by default."""
        assert checker.check_regex("ABC", r"^[A-Z]+$") is True
        assert checker.check_regex("abc", r"^[A-Z]+$") is False

    def test_empty_string(self, checker):
        """Test empty string matching."""
        assert checker.check_regex("", r"^$") is True
        assert checker.check_regex("", r"^.+$") is False

    def test_invalid_regex(self, checker):
        """Test invalid regex pattern returns False."""
        assert checker.check_regex("test", r"[invalid(") is False


# ============================================================================
# TestCheckDomain
# ============================================================================


class TestCheckDomain:
    """Test domain (allowed values) validation."""

    def test_in_set(self, checker):
        """Test value in allowed set passes."""
        assert checker.check_domain("red", ["red", "green", "blue"]) is True

    def test_not_in_set(self, checker):
        """Test value not in allowed set fails."""
        assert checker.check_domain("purple", ["red", "green", "blue"]) is False

    def test_case_sensitivity(self, checker):
        """Test domain check is case sensitive."""
        assert checker.check_domain("Red", ["red", "green", "blue"]) is False

    def test_empty_set(self, checker):
        """Test empty allowed set always fails."""
        assert checker.check_domain("anything", []) is False

    def test_none_value(self, checker):
        """Test None value not in allowed set."""
        assert checker.check_domain(None, ["red", "green"]) is False

    def test_single_value_set(self, checker):
        """Test single value domain."""
        assert checker.check_domain("only", ["only"]) is True
        assert checker.check_domain("other", ["only"]) is False


# ============================================================================
# TestCheckCrossField
# ============================================================================


class TestCheckCrossField:
    """Test cross-field constraint validation."""

    def test_start_less_than_end(self, checker):
        """Test start < end constraint passes."""
        record = {"start": 10, "end": 20}
        constraint = {"field_a": "start", "operator": "LESS_THAN", "field_b": "end"}
        assert checker.check_cross_field(record, constraint) is True

    def test_start_not_less_than_end(self, checker):
        """Test start >= end constraint fails."""
        record = {"start": 30, "end": 20}
        constraint = {"field_a": "start", "operator": "LESS_THAN", "field_b": "end"}
        assert checker.check_cross_field(record, constraint) is False

    def test_amount_greater_than(self, checker):
        """Test amount > threshold constraint."""
        record = {"amount": 100, "threshold": 50}
        constraint = {"field_a": "amount", "operator": "GREATER_THAN", "field_b": "threshold"}
        assert checker.check_cross_field(record, constraint) is True

    def test_missing_field(self, checker):
        """Test missing field returns True (cannot evaluate)."""
        record = {"start": 10}
        constraint = {"field_a": "start", "operator": "LESS_THAN", "field_b": "end"}
        assert checker.check_cross_field(record, constraint) is True

    def test_equal_fields(self, checker):
        """Test EQUALS constraint."""
        record = {"a": 10, "b": 10}
        constraint = {"field_a": "a", "operator": "EQUALS", "field_b": "b"}
        assert checker.check_cross_field(record, constraint) is True

    def test_not_equal_fields(self, checker):
        """Test NOT_EQUALS constraint."""
        record = {"a": 10, "b": 20}
        constraint = {"field_a": "a", "operator": "NOT_EQUALS", "field_b": "b"}
        assert checker.check_cross_field(record, constraint) is True

    def test_custom_constraint_unsupported(self, checker):
        """Test unsupported operator returns True."""
        record = {"a": 10, "b": 20}
        constraint = {"field_a": "a", "operator": "CUSTOM_OP", "field_b": "b"}
        assert checker.check_cross_field(record, constraint) is True

    def test_both_fields_none(self, checker):
        """Test both fields None returns True."""
        record = {"a": None, "b": None}
        constraint = {"field_a": "a", "operator": "EQUALS", "field_b": "b"}
        assert checker.check_cross_field(record, constraint) is True


# ============================================================================
# TestComputeValidityScore
# ============================================================================


class TestComputeValidityScore:
    """Test overall validity score computation."""

    def test_all_valid(self, checker):
        """Test all valid = 1.0."""
        data = [{"email": "a@b.com"}, {"email": "c@d.org"}]
        rules = [{"column": "email", "format_type": "email"}]
        result = checker.compute_validity_score(data, rules)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_all_invalid(self, checker):
        """Test all invalid = 0.0."""
        data = [{"email": "invalid"}, {"email": "also-invalid"}]
        rules = [{"column": "email", "format_type": "email"}]
        result = checker.compute_validity_score(data, rules)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_mixed(self, checker):
        """Test mixed validity score between 0 and 1."""
        data = [{"email": "a@b.com"}, {"email": "invalid"}, {"email": "c@d.org"}]
        rules = [{"column": "email", "format_type": "email"}]
        result = checker.compute_validity_score(data, rules)
        assert 0.0 < result < 1.0

    def test_empty_data(self, checker):
        """Test empty data returns 1.0."""
        result = checker.compute_validity_score([], [])
        assert result == pytest.approx(1.0, abs=0.01)

    def test_no_rules(self, checker):
        """Test no rules returns 1.0."""
        data = [{"x": 1}]
        result = checker.compute_validity_score(data, None)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_single_row(self, checker):
        """Test single row validity."""
        data = [{"email": "a@b.com"}]
        rules = [{"column": "email", "format_type": "email"}]
        result = checker.compute_validity_score(data, rules)
        assert result == pytest.approx(1.0, abs=0.01)


# ============================================================================
# TestGenerateValidityIssues
# ============================================================================


class TestGenerateValidityIssues:
    """Test validity issue generation."""

    def test_type_mismatch_issues(self, checker):
        """Test type mismatch issues generated."""
        column_results = {
            "email": {
                "invalid_count": 2,
                "total_count": 10,
                "validity_rate": 0.8,
            }
        }
        issues = checker.generate_validity_issues(column_results, [], 0.8)
        assert len(issues) > 0

    def test_format_issues(self, checker):
        """Test format violation issues generated."""
        column_results = {
            "col": {
                "invalid_count": 5,
                "total_count": 10,
                "validity_rate": 0.5,
            }
        }
        issues = checker.generate_validity_issues(column_results, [], 0.5)
        col_issues = [i for i in issues if i.get("column") == "col"]
        assert len(col_issues) > 0

    def test_no_issues_for_valid_data(self, checker):
        """Test no issues for fully valid data."""
        column_results = {
            "col": {
                "invalid_count": 0,
                "total_count": 10,
                "validity_rate": 1.0,
            }
        }
        issues = checker.generate_validity_issues(column_results, [], 1.0)
        # Should not have per-column issues or dataset-level issue
        assert all(i.get("column") != "col" for i in issues)

    def test_severity(self, checker):
        """Test severity classification in issues."""
        column_results = {
            "col": {
                "invalid_count": 6,
                "total_count": 10,
                "validity_rate": 0.4,
            }
        }
        issues = checker.generate_validity_issues(column_results, [], 0.4)
        severities = [i["severity"] for i in issues]
        assert any(s in (SEVERITY_CRITICAL, SEVERITY_HIGH) for s in severities)

    def test_description(self, checker):
        """Test issues contain descriptive messages."""
        column_results = {
            "email": {
                "invalid_count": 3,
                "total_count": 10,
                "validity_rate": 0.7,
            }
        }
        issues = checker.generate_validity_issues(column_results, [], 0.7)
        for issue in issues:
            assert "message" in issue
            assert len(issue["message"]) > 0

    def test_column_reference(self, checker):
        """Test issues reference their column."""
        column_results = {
            "my_col": {
                "invalid_count": 1,
                "total_count": 5,
                "validity_rate": 0.8,
            }
        }
        issues = checker.generate_validity_issues(column_results, [], 0.8)
        col_issues = [i for i in issues if i.get("column") == "my_col"]
        assert len(col_issues) > 0

    def test_dataset_level_issue(self, checker):
        """Test dataset-level issue when score < 0.9."""
        issues = checker.generate_validity_issues({}, [], 0.5)
        dataset_issues = [i for i in issues if i.get("column") == "__dataset__"]
        assert len(dataset_issues) == 1

    def test_no_dataset_issue_when_high_score(self, checker):
        """Test no dataset-level issue when score >= 0.9."""
        issues = checker.generate_validity_issues({}, [], 0.95)
        dataset_issues = [i for i in issues if i.get("column") == "__dataset__"]
        assert len(dataset_issues) == 0


# ============================================================================
# TestStatistics
# ============================================================================


class TestStatistics:
    """Test aggregate validation statistics."""

    def test_initial(self, checker):
        """Test initial statistics are zeroed."""
        stats = checker.get_statistics()
        assert stats["validations_completed"] == 0

    def test_post_validation(self, checker, valid_email_data):
        """Test statistics updated after validation."""
        rules = [{"column": "email", "format_type": "email"}]
        checker.validate(valid_email_data, rules=rules)
        stats = checker.get_statistics()
        assert stats["validations_completed"] == 1
        assert stats["total_rows_validated"] == 3

    def test_accumulates(self, checker, valid_email_data):
        """Test statistics accumulate."""
        rules = [{"column": "email", "format_type": "email"}]
        checker.validate(valid_email_data, rules=rules)
        checker.validate(valid_email_data, rules=rules)
        stats = checker.get_statistics()
        assert stats["validations_completed"] == 2

    def test_violations_tracked(self, checker, mixed_email_data):
        """Test violations are tracked in statistics."""
        rules = [{"column": "email", "format_type": "email"}]
        checker.validate(mixed_email_data, rules=rules)
        stats = checker.get_statistics()
        assert stats["total_violations"] > 0


# ============================================================================
# TestProvenance
# ============================================================================


class TestProvenance:
    """Test provenance hash generation."""

    def test_sha256_length(self, checker, valid_email_data):
        """Test provenance hash is 64-char SHA-256."""
        rules = [{"column": "email", "format_type": "email"}]
        result = checker.validate(valid_email_data, rules=rules)
        assert len(result["provenance_hash"]) == 64

    def test_different_data(self, checker):
        """Test different data produces different hashes."""
        data_a = [{"x": "a@b.com"}]
        data_b = [{"x": "c@d.com"}]
        rules = [{"column": "x", "format_type": "email"}]
        result_a = checker.validate(data_a, rules=rules)
        result_b = checker.validate(data_b, rules=rules)
        assert result_a["provenance_hash"] != result_b["provenance_hash"]

    def test_column_provenance(self, checker, valid_email_data):
        """Test per-column results have provenance hash."""
        rules = [{"column": "email", "format_type": "email"}]
        result = checker.validate(valid_email_data, rules=rules)
        for col, col_result in result["column_validity"].items():
            assert len(col_result["provenance_hash"]) == 64


# ============================================================================
# TestThreadSafety
# ============================================================================


class TestThreadSafety:
    """Test thread safety."""

    def test_concurrent_validation(self, checker):
        """Test multiple threads can validate concurrently."""
        errors = []
        data = [{"email": f"user{i}@test.com"} for i in range(100)]
        rules = [{"column": "email", "format_type": "email"}]

        def do_validate(_):
            try:
                checker.validate(data, rules=rules)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=do_validate, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = checker.get_statistics()
        assert stats["validations_completed"] == 10

    def test_stats_consistency(self, checker):
        """Test stats remain consistent under concurrent access."""
        data = [{"x": "a@b.com"}]
        rules = [{"column": "x", "format_type": "email"}]

        def do_validate(_):
            checker.validate(data, rules=rules)

        threads = [threading.Thread(target=do_validate, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = checker.get_statistics()
        assert stats["validations_completed"] == 5


# ============================================================================
# TestStorageAndRetrieval
# ============================================================================


class TestStorageAndRetrieval:
    """Test validation storage, retrieval, and deletion."""

    def test_get_existing(self, checker, valid_email_data):
        """Test retrieving existing validation."""
        rules = [{"column": "email", "format_type": "email"}]
        result = checker.validate(valid_email_data, rules=rules)
        retrieved = checker.get_validation(result["validation_id"])
        assert retrieved is not None
        assert retrieved["validation_id"] == result["validation_id"]

    def test_get_nonexistent(self, checker):
        """Test retrieving non-existent validation returns None."""
        assert checker.get_validation("VLD-doesnotexist") is None

    def test_list_validations(self, checker, valid_email_data):
        """Test listing validations."""
        rules = [{"column": "email", "format_type": "email"}]
        checker.validate(valid_email_data, rules=rules)
        checker.validate(valid_email_data, rules=rules)
        validations = checker.list_validations()
        assert len(validations) == 2

    def test_delete_existing(self, checker, valid_email_data):
        """Test deleting existing validation."""
        rules = [{"column": "email", "format_type": "email"}]
        result = checker.validate(valid_email_data, rules=rules)
        assert checker.delete_validation(result["validation_id"]) is True
        assert checker.get_validation(result["validation_id"]) is None

    def test_delete_nonexistent(self, checker):
        """Test deleting non-existent validation returns False."""
        assert checker.delete_validation("VLD-doesnotexist") is False


# ============================================================================
# TestHelperFunctions
# ============================================================================


class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_luhn_check_valid(self):
        """Test Luhn check with valid card number."""
        # Visa test number
        assert _luhn_check("4111111111111111") is True

    def test_luhn_check_invalid(self):
        """Test Luhn check with invalid card number."""
        assert _luhn_check("4111111111111112") is False

    def test_luhn_check_short(self):
        """Test Luhn check with too short number."""
        assert _luhn_check("1") is False

    def test_is_integer_native(self):
        """Test _is_integer with native int."""
        assert _is_integer(42) is True

    def test_is_integer_bool(self):
        """Test _is_integer rejects booleans."""
        assert _is_integer(True) is False

    def test_is_integer_string(self):
        """Test _is_integer with string int."""
        assert _is_integer("42") is True

    def test_is_integer_invalid(self):
        """Test _is_integer with non-integer string."""
        assert _is_integer("hello") is False

    def test_is_float_native(self):
        """Test _is_float with native float."""
        assert _is_float(3.14) is True

    def test_is_float_int(self):
        """Test _is_float accepts int."""
        assert _is_float(42) is True

    def test_is_float_bool(self):
        """Test _is_float rejects booleans."""
        assert _is_float(True) is False

    def test_is_float_invalid(self):
        """Test _is_float with non-numeric string."""
        assert _is_float("hello") is False

    def test_is_boolean_native(self):
        """Test _is_boolean with native bool."""
        assert _is_boolean(True) is True
        assert _is_boolean(False) is True

    def test_is_boolean_strings(self):
        """Test _is_boolean with string representations."""
        assert _is_boolean("true") is True
        assert _is_boolean("false") is True
        assert _is_boolean("yes") is True
        assert _is_boolean("no") is True

    def test_is_boolean_invalid(self):
        """Test _is_boolean with non-boolean string."""
        assert _is_boolean("hello") is False

    def test_is_string_valid(self):
        """Test _is_string with valid string."""
        assert _is_string("hello") is True

    def test_is_string_empty(self):
        """Test _is_string with empty/whitespace string."""
        assert _is_string("") is False
        assert _is_string("   ") is False

    def test_is_string_non_string(self):
        """Test _is_string with non-string value."""
        assert _is_string(42) is False


# ============================================================================
# TestValidationId
# ============================================================================


class TestValidationId:
    """Test validation ID generation."""

    def test_validation_id_prefix(self, checker, valid_email_data):
        """Test validation_id starts with VLD- prefix."""
        rules = [{"column": "email", "format_type": "email"}]
        result = checker.validate(valid_email_data, rules=rules)
        assert result["validation_id"].startswith("VLD-")

    def test_unique_ids(self, checker, valid_email_data):
        """Test each validation gets a unique ID."""
        rules = [{"column": "email", "format_type": "email"}]
        result_a = checker.validate(valid_email_data, rules=rules)
        result_b = checker.validate(valid_email_data, rules=rules)
        assert result_a["validation_id"] != result_b["validation_id"]


# ============================================================================
# TestCreditCardFormat
# ============================================================================


class TestCreditCardFormat:
    """Test credit card format with Luhn validation."""

    def test_valid_visa(self, checker):
        """Test valid Visa card passes."""
        assert checker.check_format("4111111111111111", "credit_card") is True

    def test_invalid_luhn(self, checker):
        """Test invalid Luhn fails."""
        assert checker.check_format("4111111111111112", "credit_card") is False

    def test_too_short(self, checker):
        """Test too short number fails."""
        assert checker.check_format("41111", "credit_card") is False

    def test_non_numeric(self, checker):
        """Test non-numeric string fails."""
        assert checker.check_format("abcdefghijklm", "credit_card") is False


# ============================================================================
# TestSeverityClassification
# ============================================================================


class TestSeverityClassification:
    """Test severity classification for error rates."""

    def test_critical_severity(self, checker):
        """Test critical severity for >= 50% error rate."""
        issues = checker.generate_validity_issues(
            {"col": {"invalid_count": 6, "total_count": 10, "validity_rate": 0.4}},
            [], 0.4,
        )
        col_issues = [i for i in issues if i.get("column") == "col"]
        assert col_issues[0]["severity"] == SEVERITY_CRITICAL

    def test_high_severity(self, checker):
        """Test high severity for >= 30% error rate."""
        issues = checker.generate_validity_issues(
            {"col": {"invalid_count": 4, "total_count": 10, "validity_rate": 0.6}},
            [], 0.6,
        )
        col_issues = [i for i in issues if i.get("column") == "col"]
        assert col_issues[0]["severity"] == SEVERITY_HIGH

    def test_medium_severity(self, checker):
        """Test medium severity for >= 10% error rate."""
        issues = checker.generate_validity_issues(
            {"col": {"invalid_count": 2, "total_count": 10, "validity_rate": 0.8}},
            [], 0.8,
        )
        col_issues = [i for i in issues if i.get("column") == "col"]
        assert col_issues[0]["severity"] == SEVERITY_MEDIUM

    def test_low_severity(self, checker):
        """Test low severity for < 10% error rate."""
        issues = checker.generate_validity_issues(
            {"col": {"invalid_count": 1, "total_count": 100, "validity_rate": 0.99}},
            [], 0.99,
        )
        col_issues = [i for i in issues if i.get("column") == "col"]
        assert col_issues[0]["severity"] == SEVERITY_LOW
