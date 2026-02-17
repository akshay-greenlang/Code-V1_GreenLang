# -*- coding: utf-8 -*-
"""
Unit tests for ComparisonEngine - AGENT-DATA-015

Tests all public methods of ComparisonEngine with 70+ test cases.
Validates numeric, string, date, boolean, categorical, currency,
unit-value comparisons, batch processing, summary, severity
classification, and provenance tracking.

Author: GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

import math
import pytest
from datetime import date, datetime

from greenlang.cross_source_reconciliation.comparison_engine import (
    ComparisonEngine,
    FieldComparison,
    ComparisonResult,
    ComparisonSummary,
    DiscrepancySeverity,
    FieldType,
    ToleranceRule,
    UNIT_CONVERSIONS,
    _is_missing,
    _to_float,
    _to_bool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a fresh ComparisonEngine for each test."""
    return ComparisonEngine()


@pytest.fixture
def exchange_rates():
    """Standard exchange rates for currency tests (to USD)."""
    return {
        "USD": 1.0,
        "EUR": 1.10,
        "GBP": 1.27,
        "JPY": 0.0067,
        "CHF": 1.12,
    }


@pytest.fixture
def synonym_map():
    """Standard synonym mapping for categorical tests."""
    return {
        "electricity": ["electric", "power", "elec"],
        "natural_gas": ["gas", "nat_gas", "methane"],
        "diesel": ["diesel_fuel", "gasoil"],
    }


# ---------------------------------------------------------------------------
# TestComparisonEngine: compare_numeric
# ---------------------------------------------------------------------------


class TestCompareNumeric:
    """Tests for compare_numeric method."""

    def test_exact_match(self, engine):
        """Identical numeric values produce MATCH status."""
        fc = engine.compare_numeric(100.0, 100.0)
        assert fc.status == ComparisonResult.MATCH.value

    def test_exact_match_integers(self, engine):
        """Integer comparison with exact match."""
        fc = engine.compare_numeric(42, 42)
        assert fc.status == ComparisonResult.MATCH.value

    def test_within_absolute_tolerance(self, engine):
        """Values within absolute tolerance produce WITHIN_TOLERANCE."""
        fc = engine.compare_numeric(100.0, 102.0, tolerance_abs=5.0)
        assert fc.status == ComparisonResult.WITHIN_TOLERANCE.value

    def test_within_percentage_tolerance(self, engine):
        """Values within percentage tolerance produce WITHIN_TOLERANCE."""
        fc = engine.compare_numeric(100.0, 105.0, tolerance_pct=10.0)
        assert fc.status == ComparisonResult.WITHIN_TOLERANCE.value

    def test_exceeds_absolute_tolerance(self, engine):
        """Values exceeding absolute tolerance produce MISMATCH."""
        fc = engine.compare_numeric(100.0, 110.0, tolerance_abs=5.0)
        assert fc.status == ComparisonResult.MISMATCH.value

    def test_exceeds_percentage_tolerance(self, engine):
        """Values exceeding percentage tolerance produce MISMATCH."""
        fc = engine.compare_numeric(100.0, 200.0, tolerance_pct=5.0)
        assert fc.status == ComparisonResult.MISMATCH.value

    def test_both_tolerances_must_pass(self, engine):
        """Both absolute and percentage tolerance must be satisfied."""
        # abs_diff=10, within abs(15), but rel_diff~9.5%, exceeds pct(5%)
        fc = engine.compare_numeric(
            100.0, 110.0, tolerance_abs=15.0, tolerance_pct=5.0,
        )
        assert fc.status == ComparisonResult.MISMATCH.value

    def test_absolute_diff_computed(self, engine):
        """Absolute difference is correctly computed."""
        fc = engine.compare_numeric(100.0, 107.0, tolerance_abs=10.0)
        assert fc.absolute_diff == pytest.approx(7.0, abs=1e-10)

    def test_relative_diff_computed(self, engine):
        """Relative difference percentage is correctly computed."""
        fc = engine.compare_numeric(100.0, 110.0)
        assert fc.relative_diff_pct is not None
        assert fc.relative_diff_pct > 0

    def test_missing_a_value(self, engine):
        """None value in source A produces MISSING_LEFT."""
        fc = engine.compare_numeric(None, 100.0)
        assert fc.status == ComparisonResult.MISSING_LEFT.value

    def test_missing_b_value(self, engine):
        """None value in source B produces MISSING_RIGHT."""
        fc = engine.compare_numeric(100.0, None)
        assert fc.status == ComparisonResult.MISSING_RIGHT.value

    def test_both_missing(self, engine):
        """Both None produces MISSING_BOTH."""
        fc = engine.compare_numeric(None, None)
        assert fc.status == ComparisonResult.MISSING_BOTH.value

    def test_empty_string_is_missing(self, engine):
        """Empty string is treated as missing."""
        fc = engine.compare_numeric("", 100.0)
        assert fc.status == ComparisonResult.MISSING_LEFT.value

    def test_nan_is_missing(self, engine):
        """NaN value is treated as missing."""
        fc = engine.compare_numeric(float("nan"), 100.0)
        assert fc.status == ComparisonResult.MISSING_LEFT.value

    def test_string_numeric_parsed(self, engine):
        """String numeric values are parsed to float."""
        fc = engine.compare_numeric("100.5", "100.5")
        assert fc.status == ComparisonResult.MATCH.value

    def test_unparseable_string_incomparable(self, engine):
        """Non-numeric string produces INCOMPARABLE."""
        fc = engine.compare_numeric("not_a_number", 100.0)
        assert fc.status == ComparisonResult.INCOMPARABLE.value

    def test_rounding_before_comparison(self, engine):
        """Rounding before comparison eliminates trivial differences."""
        fc = engine.compare_numeric(100.456, 100.457, rounding_digits=2)
        assert fc.status == ComparisonResult.MATCH.value

    def test_zero_values_match(self, engine):
        """Two zero values match exactly."""
        fc = engine.compare_numeric(0.0, 0.0)
        assert fc.status == ComparisonResult.MATCH.value

    def test_negative_values(self, engine):
        """Negative numeric values are compared correctly."""
        fc = engine.compare_numeric(-100.0, -100.0)
        assert fc.status == ComparisonResult.MATCH.value

    def test_provenance_hash_populated(self, engine):
        """Numeric comparison produces a provenance hash."""
        fc = engine.compare_numeric(100.0, 200.0)
        assert fc.provenance_hash != ""

    def test_field_name_stored(self, engine):
        """Field name is stored in the comparison result."""
        fc = engine.compare_numeric(100.0, 200.0, field_name="total_spend")
        assert fc.field_name == "total_spend"


# ---------------------------------------------------------------------------
# TestComparisonEngine: compare_string
# ---------------------------------------------------------------------------


class TestCompareString:
    """Tests for compare_string method."""

    def test_exact_match(self, engine):
        """Identical strings produce MATCH status."""
        fc = engine.compare_string("hello", "hello")
        assert fc.status == ComparisonResult.MATCH.value

    def test_case_insensitive_match(self, engine):
        """Case-insensitive comparison matches different cases."""
        fc = engine.compare_string("Hello World", "hello world", case_sensitive=False)
        assert fc.status == ComparisonResult.MATCH.value

    def test_case_sensitive_mismatch(self, engine):
        """Case-sensitive comparison detects case differences."""
        fc = engine.compare_string("Hello", "hello", case_sensitive=True)
        assert fc.status == ComparisonResult.MISMATCH.value

    def test_whitespace_stripping(self, engine):
        """Whitespace stripping normalizes leading/trailing spaces."""
        fc = engine.compare_string("  test  ", "test", strip_whitespace=True)
        assert fc.status == ComparisonResult.MATCH.value

    def test_whitespace_preserved(self, engine):
        """With strip_whitespace=False, spaces are significant."""
        fc = engine.compare_string(
            "  test  ", "test",
            strip_whitespace=False, case_sensitive=True,
        )
        assert fc.status == ComparisonResult.MISMATCH.value

    def test_missing_a_value(self, engine):
        """None in source A produces MISSING_LEFT."""
        fc = engine.compare_string(None, "hello")
        assert fc.status == ComparisonResult.MISSING_LEFT.value

    def test_missing_b_value(self, engine):
        """None in source B produces MISSING_RIGHT."""
        fc = engine.compare_string("hello", None)
        assert fc.status == ComparisonResult.MISSING_RIGHT.value

    def test_both_missing(self, engine):
        """Both None produces MISSING_BOTH."""
        fc = engine.compare_string(None, None)
        assert fc.status == ComparisonResult.MISSING_BOTH.value

    def test_empty_string_is_missing(self, engine):
        """Empty string is treated as missing value."""
        fc = engine.compare_string("", "hello")
        assert fc.status == ComparisonResult.MISSING_LEFT.value

    def test_string_mismatch(self, engine):
        """Different strings produce MISMATCH."""
        fc = engine.compare_string("alpha", "beta")
        assert fc.status == ComparisonResult.MISMATCH.value


# ---------------------------------------------------------------------------
# TestComparisonEngine: compare_date
# ---------------------------------------------------------------------------


class TestCompareDate:
    """Tests for compare_date method."""

    def test_same_date_strings(self, engine):
        """Identical date strings produce MATCH."""
        fc = engine.compare_date("2025-01-15", "2025-01-15")
        assert fc.status == ComparisonResult.MATCH.value

    def test_same_day_different_formats(self, engine):
        """Same date in different formats still matches."""
        fc = engine.compare_date(
            "2025-01-15", "15/01/2025",
        )
        assert fc.status == ComparisonResult.MATCH.value

    def test_within_day_tolerance(self, engine):
        """Dates within max_days_diff produce WITHIN_TOLERANCE."""
        fc = engine.compare_date("2025-01-15", "2025-01-17", max_days_diff=3)
        assert fc.status == ComparisonResult.WITHIN_TOLERANCE.value
        assert fc.absolute_diff == 2.0

    def test_exceeds_day_tolerance(self, engine):
        """Dates exceeding max_days_diff produce MISMATCH."""
        fc = engine.compare_date("2025-01-15", "2025-02-15", max_days_diff=3)
        assert fc.status == ComparisonResult.MISMATCH.value

    def test_exact_match_zero_tolerance(self, engine):
        """Zero tolerance requires exact date match."""
        fc = engine.compare_date("2025-01-15", "2025-01-16", max_days_diff=0)
        assert fc.status == ComparisonResult.MISMATCH.value

    def test_date_objects_as_input(self, engine):
        """Python date objects are accepted as input."""
        fc = engine.compare_date(date(2025, 1, 15), date(2025, 1, 15))
        assert fc.status == ComparisonResult.MATCH.value

    def test_datetime_objects_as_input(self, engine):
        """Python datetime objects are accepted."""
        fc = engine.compare_date(
            datetime(2025, 1, 15, 10, 30),
            datetime(2025, 1, 15, 14, 45),
        )
        assert fc.status == ComparisonResult.MATCH.value  # Same day

    def test_missing_date_a(self, engine):
        """None date A produces MISSING_LEFT."""
        fc = engine.compare_date(None, "2025-01-15")
        assert fc.status == ComparisonResult.MISSING_LEFT.value

    def test_missing_date_b(self, engine):
        """None date B produces MISSING_RIGHT."""
        fc = engine.compare_date("2025-01-15", None)
        assert fc.status == ComparisonResult.MISSING_RIGHT.value

    def test_unparseable_date(self, engine):
        """Unparseable date string produces INCOMPARABLE."""
        fc = engine.compare_date("not-a-date", "2025-01-15")
        assert fc.status == ComparisonResult.INCOMPARABLE.value

    def test_specified_date_format(self, engine):
        """Specified date format is used for parsing."""
        fc = engine.compare_date(
            "15-01-2025", "15-01-2025",
            date_format_a="%d-%m-%Y",
            date_format_b="%d-%m-%Y",
        )
        assert fc.status == ComparisonResult.MATCH.value


# ---------------------------------------------------------------------------
# TestComparisonEngine: compare_boolean
# ---------------------------------------------------------------------------


class TestCompareBoolean:
    """Tests for compare_boolean method."""

    def test_true_true_match(self, engine):
        """True/True produces MATCH."""
        fc = engine.compare_boolean(True, True)
        assert fc.status == ComparisonResult.MATCH.value

    def test_false_false_match(self, engine):
        """False/False produces MATCH."""
        fc = engine.compare_boolean(False, False)
        assert fc.status == ComparisonResult.MATCH.value

    def test_true_false_mismatch(self, engine):
        """True/False produces MISMATCH."""
        fc = engine.compare_boolean(True, False)
        assert fc.status == ComparisonResult.MISMATCH.value

    @pytest.mark.parametrize("truthy_val", ["true", "yes", "1", "y", "on", "t"])
    def test_truthy_string_matches_true(self, engine, truthy_val):
        """Truthy string values match boolean True."""
        fc = engine.compare_boolean(truthy_val, True)
        assert fc.status == ComparisonResult.MATCH.value

    @pytest.mark.parametrize("falsy_val", ["false", "no", "0", "n", "off", "f"])
    def test_falsy_string_matches_false(self, engine, falsy_val):
        """Falsy string values match boolean False."""
        fc = engine.compare_boolean(falsy_val, False)
        assert fc.status == ComparisonResult.MATCH.value

    def test_yes_true_match(self, engine):
        """'yes' matches True."""
        fc = engine.compare_boolean("yes", True)
        assert fc.status == ComparisonResult.MATCH.value

    def test_no_false_match(self, engine):
        """'no' matches False."""
        fc = engine.compare_boolean("no", False)
        assert fc.status == ComparisonResult.MATCH.value

    def test_numeric_one_is_truthy(self, engine):
        """Numeric 1 is truthy, 0 is falsy."""
        fc = engine.compare_boolean(1, True)
        assert fc.status == ComparisonResult.MATCH.value
        fc2 = engine.compare_boolean(0, False)
        assert fc2.status == ComparisonResult.MATCH.value

    def test_unparseable_boolean(self, engine):
        """Unparseable boolean value produces INCOMPARABLE."""
        fc = engine.compare_boolean("maybe", True)
        assert fc.status == ComparisonResult.INCOMPARABLE.value

    def test_missing_boolean(self, engine):
        """None boolean value produces MISSING status."""
        fc = engine.compare_boolean(None, True)
        assert fc.status == ComparisonResult.MISSING_LEFT.value


# ---------------------------------------------------------------------------
# TestComparisonEngine: compare_categorical
# ---------------------------------------------------------------------------


class TestCompareCategorical:
    """Tests for compare_categorical method."""

    def test_direct_match(self, engine):
        """Identical categorical values produce MATCH."""
        fc = engine.compare_categorical("electricity", "electricity")
        assert fc.status == ComparisonResult.MATCH.value

    def test_case_insensitive_match(self, engine):
        """Categorical comparison is case-insensitive."""
        fc = engine.compare_categorical("Electricity", "electricity")
        assert fc.status == ComparisonResult.MATCH.value

    def test_synonym_match(self, engine, synonym_map):
        """Synonymous values produce MATCH."""
        fc = engine.compare_categorical(
            "electricity", "power", synonyms=synonym_map,
        )
        assert fc.status == ComparisonResult.MATCH.value

    def test_synonym_match_reverse(self, engine, synonym_map):
        """Synonym matching works in both directions."""
        fc = engine.compare_categorical(
            "electric", "electricity", synonyms=synonym_map,
        )
        assert fc.status == ComparisonResult.MATCH.value

    def test_no_synonym_mismatch(self, engine, synonym_map):
        """Non-synonymous values produce MISMATCH."""
        fc = engine.compare_categorical(
            "electricity", "diesel", synonyms=synonym_map,
        )
        assert fc.status == ComparisonResult.MISMATCH.value

    def test_no_synonyms_provided(self, engine):
        """Without synonyms, only direct match works."""
        fc = engine.compare_categorical("electricity", "power")
        assert fc.status == ComparisonResult.MISMATCH.value

    def test_missing_categorical(self, engine):
        """None categorical value produces MISSING status."""
        fc = engine.compare_categorical(None, "diesel")
        assert fc.status == ComparisonResult.MISSING_LEFT.value


# ---------------------------------------------------------------------------
# TestComparisonEngine: compare_currency
# ---------------------------------------------------------------------------


class TestCompareCurrency:
    """Tests for compare_currency method."""

    def test_same_currency_match(self, engine, exchange_rates):
        """Same currency with equal values produces MATCH."""
        fc = engine.compare_currency(
            100.0, 100.0, "USD", "USD", exchange_rates,
        )
        assert fc.status == ComparisonResult.MATCH.value

    def test_different_currencies_with_conversion(self, engine, exchange_rates):
        """Different currencies are converted before comparison."""
        # 100 EUR = 110 USD, 110 USD = 110 USD -> match
        fc = engine.compare_currency(
            100.0, 110.0, "EUR", "USD", exchange_rates, tolerance_pct=1.0,
        )
        assert fc.status == ComparisonResult.MATCH.value

    def test_currency_conversion_mismatch(self, engine, exchange_rates):
        """Currency values that differ after conversion produce MISMATCH."""
        # 100 EUR = 110 USD vs 200 USD -> big difference
        fc = engine.compare_currency(
            100.0, 200.0, "EUR", "USD", exchange_rates, tolerance_pct=1.0,
        )
        assert fc.status == ComparisonResult.MISMATCH.value

    def test_same_currency_within_tolerance(self, engine, exchange_rates):
        """Same currency within percentage tolerance."""
        fc = engine.compare_currency(
            100.0, 103.0, "USD", "USD", exchange_rates, tolerance_pct=5.0,
        )
        assert fc.status == ComparisonResult.WITHIN_TOLERANCE.value

    def test_missing_exchange_rate(self, engine):
        """Missing exchange rate produces INCOMPARABLE."""
        fc = engine.compare_currency(
            100.0, 100.0, "XYZ", "USD", {"USD": 1.0},
        )
        assert fc.status == ComparisonResult.INCOMPARABLE.value

    def test_missing_currency_value(self, engine, exchange_rates):
        """None currency value produces MISSING status."""
        fc = engine.compare_currency(
            None, 100.0, "USD", "USD", exchange_rates,
        )
        assert fc.status == ComparisonResult.MISSING_LEFT.value

    def test_both_missing_currency(self, engine, exchange_rates):
        """Both None produces MISSING_BOTH."""
        fc = engine.compare_currency(
            None, None, "USD", "USD", exchange_rates,
        )
        assert fc.status == ComparisonResult.MISSING_BOTH.value


# ---------------------------------------------------------------------------
# TestComparisonEngine: compare_unit_value
# ---------------------------------------------------------------------------


class TestCompareUnitValue:
    """Tests for compare_unit_value method."""

    def test_same_unit_match(self, engine):
        """Same unit with equal values produces MATCH."""
        fc = engine.compare_unit_value(100.0, 100.0, "kg", "kg")
        assert fc.status == ComparisonResult.MATCH.value

    def test_kg_to_tonnes_conversion(self, engine):
        """1000 kg equals 1 tonne."""
        fc = engine.compare_unit_value(
            1000.0, 1.0, "kg", "tonnes", tolerance_pct=0.1,
        )
        assert fc.status in (
            ComparisonResult.MATCH.value,
            ComparisonResult.WITHIN_TOLERANCE.value,
        )

    def test_kwh_to_mwh_conversion(self, engine):
        """1000 kWh equals 1 MWh."""
        fc = engine.compare_unit_value(
            1000.0, 1.0, "kwh", "mwh", tolerance_pct=0.1,
        )
        assert fc.status in (
            ComparisonResult.MATCH.value,
            ComparisonResult.WITHIN_TOLERANCE.value,
        )

    def test_tonnes_to_kg_conversion(self, engine):
        """1 tonne equals 1000 kg."""
        fc = engine.compare_unit_value(
            1.0, 1000.0, "tonnes", "kg", tolerance_pct=0.1,
        )
        assert fc.status in (
            ComparisonResult.MATCH.value,
            ComparisonResult.WITHIN_TOLERANCE.value,
        )

    def test_different_dimension_incomparable(self, engine):
        """Units from different dimensions are INCOMPARABLE."""
        fc = engine.compare_unit_value(100.0, 100.0, "kg", "kwh")
        assert fc.status == ComparisonResult.INCOMPARABLE.value

    def test_unit_value_mismatch(self, engine):
        """Different quantities in same unit produce MISMATCH."""
        fc = engine.compare_unit_value(
            100.0, 500.0, "kg", "kg", tolerance_pct=5.0,
        )
        assert fc.status == ComparisonResult.MISMATCH.value

    def test_missing_unit_value(self, engine):
        """None value produces MISSING status."""
        fc = engine.compare_unit_value(None, 100.0, "kg", "kg")
        assert fc.status == ComparisonResult.MISSING_LEFT.value

    def test_tco2e_to_kgco2e_conversion(self, engine):
        """1 tCO2e equals 1000 kgCO2e."""
        fc = engine.compare_unit_value(
            1.0, 1000.0, "tCO2e", "kgCO2e", tolerance_pct=0.1,
        )
        assert fc.status in (
            ComparisonResult.MATCH.value,
            ComparisonResult.WITHIN_TOLERANCE.value,
        )

    def test_unit_value_within_tolerance(self, engine):
        """Values within tolerance after unit conversion."""
        fc = engine.compare_unit_value(
            1.01, 1010.0, "tonnes", "kg", tolerance_pct=1.0,
        )
        assert fc.status in (
            ComparisonResult.MATCH.value,
            ComparisonResult.WITHIN_TOLERANCE.value,
        )


# ---------------------------------------------------------------------------
# TestComparisonEngine: compare_records
# ---------------------------------------------------------------------------


class TestCompareRecords:
    """Tests for compare_records method."""

    def test_compare_records_dispatches_to_correct_methods(self, engine):
        """compare_records routes each field to the correct comparison method."""
        record_a = {"amount": 100, "name": "Acme", "active": True}
        record_b = {"amount": 102, "name": "Acme", "active": True}
        comparisons = engine.compare_records(
            record_a, record_b,
            fields=["amount", "name", "active"],
            field_types={
                "amount": FieldType.NUMERIC,
                "name": FieldType.STRING,
                "active": FieldType.BOOLEAN,
            },
            tolerance_rules={
                "amount": ToleranceRule(tolerance_pct=5.0),
            },
        )
        assert len(comparisons) == 3
        # amount should be within tolerance
        amount_comp = comparisons[0]
        assert amount_comp.field_name == "amount"
        assert amount_comp.status in (
            ComparisonResult.MATCH.value,
            ComparisonResult.WITHIN_TOLERANCE.value,
        )
        # name should match exactly
        name_comp = comparisons[1]
        assert name_comp.status == ComparisonResult.MATCH.value
        # active should match
        active_comp = comparisons[2]
        assert active_comp.status == ComparisonResult.MATCH.value

    def test_compare_records_defaults_to_string(self, engine):
        """Without field_types, fields default to STRING comparison."""
        record_a = {"x": "hello"}
        record_b = {"x": "hello"}
        comparisons = engine.compare_records(record_a, record_b, fields=["x"])
        assert len(comparisons) == 1
        assert comparisons[0].status == ComparisonResult.MATCH.value

    def test_compare_records_missing_field_in_record(self, engine):
        """Missing field in one record produces MISSING status."""
        record_a = {"x": "hello"}
        record_b = {}
        comparisons = engine.compare_records(
            record_a, record_b, fields=["x"],
        )
        assert comparisons[0].status == ComparisonResult.MISSING_RIGHT.value


# ---------------------------------------------------------------------------
# TestComparisonEngine: compare_batch
# ---------------------------------------------------------------------------


class TestCompareBatch:
    """Tests for compare_batch method."""

    def test_compare_batch_processes_all_pairs(self, engine):
        """Batch comparison processes all matched pairs."""
        pairs = [
            ({"amount": 100}, {"amount": 100}),
            ({"amount": 200}, {"amount": 210}),
            ({"amount": 300}, {"amount": 300}),
        ]
        results = engine.compare_batch(
            pairs, fields=["amount"],
            field_types={"amount": FieldType.NUMERIC},
        )
        assert len(results) == 3

    def test_compare_batch_empty_pairs(self, engine):
        """Empty pairs list produces empty results."""
        results = engine.compare_batch([], fields=["amount"])
        assert results == []

    def test_compare_batch_each_pair_has_correct_count(self, engine):
        """Each pair comparison has the correct number of field comparisons."""
        pairs = [
            ({"a": 1, "b": "x"}, {"a": 1, "b": "x"}),
        ]
        results = engine.compare_batch(
            pairs, fields=["a", "b"],
            field_types={"a": FieldType.NUMERIC, "b": FieldType.STRING},
        )
        assert len(results) == 1
        assert len(results[0]) == 2

    def test_compare_batch_with_tolerance_rules(self, engine):
        """Batch comparison applies tolerance rules."""
        pairs = [
            ({"amount": 100}, {"amount": 104}),
        ]
        results = engine.compare_batch(
            pairs, fields=["amount"],
            field_types={"amount": FieldType.NUMERIC},
            tolerance_rules={"amount": ToleranceRule(tolerance_pct=5.0)},
        )
        assert results[0][0].status == ComparisonResult.WITHIN_TOLERANCE.value


# ---------------------------------------------------------------------------
# TestComparisonEngine: summarize_comparisons
# ---------------------------------------------------------------------------


class TestSummarizeComparisons:
    """Tests for summarize_comparisons method."""

    def test_summary_counts_matches(self, engine):
        """Summary correctly counts matches."""
        fc1 = engine.compare_numeric(100, 100)
        fc2 = engine.compare_string("a", "a")
        summary = engine.summarize_comparisons([fc1, fc2])
        assert summary.total_fields == 2
        assert summary.matches == 2
        assert summary.match_rate == 1.0

    def test_summary_counts_mismatches(self, engine):
        """Summary correctly counts mismatches."""
        fc1 = engine.compare_numeric(100, 200)
        summary = engine.summarize_comparisons([fc1])
        assert summary.mismatches == 1

    def test_summary_counts_within_tolerance(self, engine):
        """Summary correctly counts within_tolerance."""
        fc = engine.compare_numeric(100, 103, tolerance_pct=5.0)
        summary = engine.summarize_comparisons([fc])
        assert summary.within_tolerance == 1

    def test_summary_counts_missing(self, engine):
        """Summary correctly counts missing values."""
        fc = engine.compare_numeric(None, 100)
        summary = engine.summarize_comparisons([fc])
        assert summary.missing == 1

    def test_summary_match_rate(self, engine):
        """Match rate is (matches + within_tolerance) / total."""
        fc1 = engine.compare_numeric(100, 100)  # match
        fc2 = engine.compare_numeric(100, 103, tolerance_pct=5.0)  # within_tol
        fc3 = engine.compare_numeric(100, 200)  # mismatch
        summary = engine.summarize_comparisons([fc1, fc2, fc3])
        expected_rate = 2 / 3
        assert summary.match_rate == pytest.approx(expected_rate, abs=1e-5)

    def test_summary_empty_list(self, engine):
        """Empty comparison list produces zero counts."""
        summary = engine.summarize_comparisons([])
        assert summary.total_fields == 0
        assert summary.match_rate == 0.0

    def test_summary_provenance_hash(self, engine):
        """Summary has a provenance hash."""
        fc = engine.compare_numeric(100, 100)
        summary = engine.summarize_comparisons([fc])
        assert summary.provenance_hash != ""

    def test_summary_severity_counts(self, engine):
        """Summary includes severity counts."""
        fc = engine.compare_numeric(100, 100)
        summary = engine.summarize_comparisons([fc])
        assert isinstance(summary.severity_counts, dict)


# ---------------------------------------------------------------------------
# TestComparisonEngine: classify_severity
# ---------------------------------------------------------------------------


class TestClassifySeverity:
    """Tests for classify_severity method."""

    def test_match_returns_none_severity(self, engine):
        """MATCH status produces NONE severity."""
        fc = engine.compare_numeric(100, 100)
        sev = engine.classify_severity(fc)
        assert sev == DiscrepancySeverity.NONE

    def test_critical_severity(self, engine):
        """Large deviation (>= 50%) is CRITICAL."""
        fc = engine.compare_numeric(100, 200)
        sev = engine.classify_severity(fc)
        assert sev == DiscrepancySeverity.CRITICAL

    def test_high_severity(self, engine):
        """Deviation >= 20% but < 50% is HIGH."""
        fc = engine.compare_numeric(100, 125)
        sev = engine.classify_severity(fc, critical_pct=50.0, high_pct=20.0)
        assert sev == DiscrepancySeverity.HIGH

    def test_medium_severity(self, engine):
        """Deviation >= 5% but < 20% is MEDIUM."""
        fc = engine.compare_numeric(100, 110)
        sev = engine.classify_severity(fc, critical_pct=50.0, high_pct=20.0, medium_pct=5.0)
        assert sev == DiscrepancySeverity.MEDIUM

    def test_low_severity(self, engine):
        """Small deviation > 0% but < 5% is LOW."""
        fc = engine.compare_numeric(100, 102)
        sev = engine.classify_severity(fc, critical_pct=50.0, high_pct=20.0, medium_pct=5.0)
        assert sev == DiscrepancySeverity.LOW

    def test_missing_left_is_medium(self, engine):
        """MISSING_LEFT produces MEDIUM severity."""
        fc = engine.compare_numeric(None, 100)
        sev = engine.classify_severity(fc)
        assert sev == DiscrepancySeverity.MEDIUM

    def test_missing_right_is_medium(self, engine):
        """MISSING_RIGHT produces MEDIUM severity."""
        fc = engine.compare_numeric(100, None)
        sev = engine.classify_severity(fc)
        assert sev == DiscrepancySeverity.MEDIUM

    def test_missing_both_is_info(self, engine):
        """MISSING_BOTH produces INFO severity."""
        fc = engine.compare_numeric(None, None)
        sev = engine.classify_severity(fc)
        assert sev == DiscrepancySeverity.INFO

    def test_incomparable_is_info(self, engine):
        """INCOMPARABLE produces INFO severity."""
        fc = engine.compare_numeric("abc", 100)
        sev = engine.classify_severity(fc)
        assert sev == DiscrepancySeverity.INFO

    def test_string_mismatch_is_medium(self, engine):
        """Non-numeric mismatch defaults to MEDIUM."""
        fc = engine.compare_string("alpha", "beta")
        sev = engine.classify_severity(fc)
        assert sev == DiscrepancySeverity.MEDIUM

    def test_custom_thresholds(self, engine):
        """Custom severity thresholds are respected."""
        fc = engine.compare_numeric(100, 108)
        # With custom thresholds: critical=30, high=15, medium=3
        sev = engine.classify_severity(fc, critical_pct=30.0, high_pct=15.0, medium_pct=3.0)
        assert sev == DiscrepancySeverity.MEDIUM


# ---------------------------------------------------------------------------
# TestComparisonEngine: Unit Conversion Tables
# ---------------------------------------------------------------------------


class TestUnitConversionTables:
    """Tests for UNIT_CONVERSIONS constant correctness."""

    def test_mass_kg_canonical(self):
        """Canonical mass unit (kg) has factor 1.0."""
        assert UNIT_CONVERSIONS["mass"]["kg"] == 1.0

    def test_mass_tonnes_factor(self):
        """1 tonne = 1000 kg."""
        assert UNIT_CONVERSIONS["mass"]["tonnes"] == 1000.0

    def test_energy_kwh_canonical(self):
        """Canonical energy unit (kWh) has factor 1.0."""
        assert UNIT_CONVERSIONS["energy"]["kwh"] == 1.0

    def test_energy_mwh_factor(self):
        """1 MWh = 1000 kWh."""
        assert UNIT_CONVERSIONS["energy"]["mwh"] == 1000.0

    def test_emissions_kgco2e_canonical(self):
        """Canonical emissions unit (kgCO2e) has factor 1.0."""
        assert UNIT_CONVERSIONS["emissions"]["kgCO2e"] == 1.0

    def test_emissions_tco2e_factor(self):
        """1 tCO2e = 1000 kgCO2e."""
        assert UNIT_CONVERSIONS["emissions"]["tCO2e"] == 1000.0

    def test_volume_liters_canonical(self):
        """Canonical volume unit (liters) has factor 1.0."""
        assert UNIT_CONVERSIONS["volume"]["liters"] == 1.0

    def test_all_dimensions_present(self):
        """All expected dimensions are present in conversions."""
        expected = {"mass", "energy", "volume", "distance", "area", "emissions"}
        assert expected.issubset(set(UNIT_CONVERSIONS.keys()))


# ---------------------------------------------------------------------------
# TestComparisonEngine: Helper Functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_is_missing_none(self):
        """None is missing."""
        assert _is_missing(None) is True

    def test_is_missing_empty_string(self):
        """Empty string is missing."""
        assert _is_missing("") is True

    def test_is_missing_whitespace_string(self):
        """Whitespace-only string is missing."""
        assert _is_missing("   ") is True

    def test_is_missing_nan(self):
        """NaN is missing."""
        assert _is_missing(float("nan")) is True

    def test_is_missing_inf(self):
        """Inf is missing."""
        assert _is_missing(float("inf")) is True

    def test_is_missing_valid_string(self):
        """Non-empty string is not missing."""
        assert _is_missing("hello") is False

    def test_is_missing_zero(self):
        """Zero is not missing."""
        assert _is_missing(0) is False

    def test_to_float_int(self):
        """Integer converts to float."""
        assert _to_float(42) == 42.0

    def test_to_float_string(self):
        """Numeric string converts to float."""
        assert _to_float("3.14") == pytest.approx(3.14)

    def test_to_float_string_with_commas(self):
        """Comma-formatted string converts to float."""
        assert _to_float("1,234.56") == pytest.approx(1234.56)

    def test_to_float_none(self):
        """None returns None."""
        assert _to_float(None) is None

    def test_to_float_unparseable(self):
        """Non-numeric string returns None."""
        assert _to_float("not_a_number") is None

    def test_to_bool_true(self):
        """True converts to True."""
        assert _to_bool(True) is True

    def test_to_bool_string_yes(self):
        """'yes' converts to True."""
        assert _to_bool("yes") is True

    def test_to_bool_string_no(self):
        """'no' converts to False."""
        assert _to_bool("no") is False

    def test_to_bool_none(self):
        """None returns None."""
        assert _to_bool(None) is None

    def test_to_bool_unparseable(self):
        """Unparseable string returns None."""
        assert _to_bool("maybe") is None
