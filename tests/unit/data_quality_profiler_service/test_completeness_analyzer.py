# -*- coding: utf-8 -*-
"""
Unit Tests for CompletenessAnalyzer Engine - AGENT-DATA-010 (GL-DATA-X-013)
===========================================================================

Tests CompletenessAnalyzer from greenlang.data_quality_profiler.completeness_analyzer.

Covers:
    - Initialization (default/custom config, stats, None config)
    - Full dataset analysis (return type, per-column, required fields, score, edge cases)
    - Per-column analysis (fully populated, all null, mixed, empty strings, whitespace)
    - Completeness score computation
    - Missing pattern detection (MCAR, MAR, MNAR, UNKNOWN)
    - Required field gap finding
    - Per-record completeness
    - Coverage matrix
    - Issue generation
    - Aggregate statistics
    - Provenance hashing
    - Thread safety

Target: 100+ tests, ~900 lines.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List

import pytest

from greenlang.data_quality_profiler.completeness_analyzer import (
    CompletenessAnalyzer,
    _is_missing,
    _safe_stdev,
    PATTERN_MCAR,
    PATTERN_MAR,
    PATTERN_MNAR,
    PATTERN_UNKNOWN,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
    SEVERITY_INFO,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def analyzer():
    """Create a CompletenessAnalyzer with default config."""
    return CompletenessAnalyzer()


@pytest.fixture
def custom_analyzer():
    """Create a CompletenessAnalyzer with custom thresholds."""
    return CompletenessAnalyzer(config={
        "critical_threshold": 0.6,
        "high_threshold": 0.4,
        "medium_threshold": 0.2,
        "mcar_stddev_threshold": 0.15,
        "mar_correlation_threshold": 0.6,
        "mnar_rate_threshold": 0.4,
    })


@pytest.fixture
def complete_data():
    """Fully complete dataset with no missing values."""
    return [
        {"name": "Alice", "age": 30, "city": "NYC"},
        {"name": "Bob", "age": 25, "city": "London"},
        {"name": "Charlie", "age": 35, "city": "Berlin"},
        {"name": "Diana", "age": 28, "city": "Tokyo"},
        {"name": "Eve", "age": 42, "city": "Paris"},
    ]


@pytest.fixture
def partial_data():
    """Dataset with mixed missing values."""
    return [
        {"name": "Alice", "age": 30, "email": "a@b.com"},
        {"name": None, "age": 25, "email": "b@c.com"},
        {"name": "Charlie", "age": None, "email": None},
        {"name": "", "age": 35, "email": "d@e.com"},
        {"name": "Eve", "age": None, "email": ""},
    ]


@pytest.fixture
def all_null_data():
    """Dataset with one entirely null column."""
    return [
        {"name": "Alice", "notes": None},
        {"name": "Bob", "notes": None},
        {"name": "Charlie", "notes": None},
        {"name": "Diana", "notes": None},
        {"name": "Eve", "notes": None},
    ]


# ============================================================================
# TestInit
# ============================================================================


class TestInit:
    """Test CompletenessAnalyzer initialization."""

    def test_default_config(self):
        """Test default thresholds are set."""
        analyzer = CompletenessAnalyzer()
        assert analyzer._critical_threshold == 0.5
        assert analyzer._high_threshold == 0.3
        assert analyzer._medium_threshold == 0.1

    def test_custom_config(self):
        """Test custom threshold overrides."""
        analyzer = CompletenessAnalyzer(config={
            "critical_threshold": 0.7,
            "high_threshold": 0.5,
        })
        assert analyzer._critical_threshold == 0.7
        assert analyzer._high_threshold == 0.5

    def test_initial_stats(self):
        """Test statistics are zeroed on init."""
        analyzer = CompletenessAnalyzer()
        stats = analyzer.get_statistics()
        assert stats["analyses_completed"] == 0
        assert stats["total_rows_analyzed"] == 0

    def test_none_config(self):
        """Test None config uses defaults."""
        analyzer = CompletenessAnalyzer(config=None)
        assert analyzer._critical_threshold == 0.5


# ============================================================================
# TestAnalyze
# ============================================================================


class TestAnalyze:
    """Test full dataset analysis."""

    def test_returns_dict(self, analyzer, complete_data):
        """Test analyze() returns a dictionary."""
        result = analyzer.analyze(complete_data)
        assert isinstance(result, dict)

    def test_per_column_completeness(self, analyzer, partial_data):
        """Test column_completeness contains results for each column."""
        result = analyzer.analyze(partial_data)
        assert "name" in result["column_completeness"]
        assert "age" in result["column_completeness"]
        assert "email" in result["column_completeness"]

    def test_required_fields(self, analyzer, partial_data):
        """Test required_field_gaps populated when required fields specified."""
        result = analyzer.analyze(partial_data, required_fields=["name", "email"])
        assert len(result["required_field_gaps"]) > 0

    def test_overall_score_complete(self, analyzer, complete_data):
        """Test overall score is 1.0 for fully complete data."""
        result = analyzer.analyze(complete_data)
        assert result["completeness_score"] == pytest.approx(1.0, abs=0.01)

    def test_empty_data_raises_error(self, analyzer):
        """Test empty data raises ValueError."""
        with pytest.raises(ValueError, match="Cannot analyse empty dataset"):
            analyzer.analyze([])

    def test_all_complete(self, analyzer, complete_data):
        """Test completeness score is 1.0 when all values present."""
        result = analyzer.analyze(complete_data)
        assert result["completeness_score"] == pytest.approx(1.0, abs=0.01)

    def test_all_null_column(self, analyzer, all_null_data):
        """Test column with all null values has fill_rate 0."""
        result = analyzer.analyze(all_null_data)
        notes_col = result["column_completeness"]["notes"]
        assert notes_col["fill_rate"] == pytest.approx(0.0, abs=0.01)

    def test_mixed_completeness(self, analyzer, partial_data):
        """Test mixed data produces score between 0 and 1."""
        result = analyzer.analyze(partial_data)
        assert 0.0 < result["completeness_score"] < 1.0

    def test_coverage_matrix_present(self, analyzer, complete_data):
        """Test coverage_matrix is present in output."""
        result = analyzer.analyze(complete_data)
        assert "coverage_matrix" in result
        assert isinstance(result["coverage_matrix"], dict)

    def test_issues_list(self, analyzer, partial_data):
        """Test issues list is returned."""
        result = analyzer.analyze(partial_data)
        assert "issues" in result
        assert isinstance(result["issues"], list)

    def test_provenance_hash(self, analyzer, complete_data):
        """Test provenance_hash is a 64-char hex string."""
        result = analyzer.analyze(complete_data)
        assert len(result["provenance_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in result["provenance_hash"])

    def test_column_scores_between_0_and_1(self, analyzer, partial_data):
        """Test all column fill_rate values are between 0.0 and 1.0."""
        result = analyzer.analyze(partial_data)
        for col_name, col_result in result["column_completeness"].items():
            assert 0.0 <= col_result["fill_rate"] <= 1.0


# ============================================================================
# TestAnalyzeColumn
# ============================================================================


class TestAnalyzeColumn:
    """Test per-column completeness analysis."""

    def test_fully_populated(self, analyzer):
        """Test fully populated column has fill_rate 1.0."""
        result = analyzer.analyze_column(["a", "b", "c", "d"], "col")
        assert result["fill_rate"] == pytest.approx(1.0, abs=0.01)
        assert result["null_count"] == 0

    def test_all_null(self, analyzer):
        """Test all null column has fill_rate 0.0."""
        result = analyzer.analyze_column([None, None, None], "col")
        assert result["fill_rate"] == pytest.approx(0.0, abs=0.01)
        assert result["null_count"] == 3

    def test_mixed(self, analyzer):
        """Test mixed column has correct null count."""
        result = analyzer.analyze_column(["a", None, "c", None, "e"], "col")
        assert result["null_count"] == 2
        assert result["fill_count"] == 3

    def test_empty_strings_counted_as_null(self, analyzer):
        """Test empty strings are treated as null."""
        result = analyzer.analyze_column(["a", "", "c", "", "e"], "col")
        assert result["null_count"] == 2

    def test_whitespace_only_counted_as_null(self, analyzer):
        """Test whitespace-only strings are treated as null."""
        result = analyzer.analyze_column(["a", "   ", "c", " \t ", "e"], "col")
        assert result["null_count"] == 2

    def test_zero_values_not_null(self, analyzer):
        """Test zero values are NOT treated as null."""
        result = analyzer.analyze_column([0, 0, 0, 0], "col")
        assert result["null_count"] == 0
        assert result["fill_rate"] == pytest.approx(1.0, abs=0.01)

    def test_false_values_not_null(self, analyzer):
        """Test False values are NOT treated as null."""
        result = analyzer.analyze_column([False, False, True], "col")
        assert result["null_count"] == 0

    def test_none_values(self, analyzer):
        """Test None values are counted as null."""
        result = analyzer.analyze_column([None, None, "data"], "col")
        assert result["null_count"] == 2

    def test_column_name_in_result(self, analyzer):
        """Test column_name is preserved in result."""
        result = analyzer.analyze_column(["a"], "my_column")
        assert result["column_name"] == "my_column"

    def test_severity_classification(self, analyzer):
        """Test severity is classified based on null rate."""
        # > 50% null -> critical
        result = analyzer.analyze_column([None, None, None, "a"], "col")
        assert result["severity"] == SEVERITY_CRITICAL


# ============================================================================
# TestComputeCompletenessScore
# ============================================================================


class TestComputeCompletenessScore:
    """Test overall completeness score computation."""

    def test_all_present(self, analyzer):
        """Test all present values = 1.0."""
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        assert analyzer.compute_completeness_score(data) == pytest.approx(1.0, abs=0.01)

    def test_all_null(self, analyzer):
        """Test all null values = 0.0."""
        data = [{"a": None, "b": None}, {"a": None, "b": None}]
        assert analyzer.compute_completeness_score(data) == pytest.approx(0.0, abs=0.01)

    def test_half_null(self, analyzer):
        """Test half null values = 0.5."""
        data = [{"a": 1, "b": None}, {"a": None, "b": 2}]
        assert analyzer.compute_completeness_score(data) == pytest.approx(0.5, abs=0.01)

    def test_varied_columns(self, analyzer):
        """Test varied completeness across columns."""
        data = [
            {"a": 1, "b": None, "c": 3},
            {"a": None, "b": 2, "c": None},
            {"a": 1, "b": 2, "c": 3},
        ]
        # 9 cells total, 3 null = 6/9 = 0.6667
        result = analyzer.compute_completeness_score(data)
        assert result == pytest.approx(6.0 / 9.0, abs=0.01)

    def test_single_row(self, analyzer):
        """Test single row completeness."""
        data = [{"a": 1, "b": None}]
        assert analyzer.compute_completeness_score(data) == pytest.approx(0.5, abs=0.01)

    def test_empty_data(self, analyzer):
        """Test empty data returns 0.0."""
        assert analyzer.compute_completeness_score([]) == pytest.approx(0.0, abs=0.01)

    def test_with_column_subset(self, analyzer):
        """Test score with column subset."""
        data = [{"a": 1, "b": None, "c": 3}]
        result = analyzer.compute_completeness_score(data, columns=["a", "c"])
        assert result == pytest.approx(1.0, abs=0.01)

    def test_whitespace_counted_as_null(self, analyzer):
        """Test whitespace-only strings reduce completeness."""
        data = [{"a": "  ", "b": "value"}]
        result = analyzer.compute_completeness_score(data)
        assert result == pytest.approx(0.5, abs=0.01)


# ============================================================================
# TestDetectMissingPattern
# ============================================================================


class TestDetectMissingPattern:
    """Test missing data pattern detection."""

    def test_mcar_uniform_missing(self, analyzer):
        """Test MCAR detected for uniformly distributed missing data."""
        data = [
            {"a": 1, "b": 2, "c": 3},
            {"a": None, "b": 2, "c": 3},
            {"a": 1, "b": None, "c": 3},
            {"a": 1, "b": 2, "c": None},
            {"a": 1, "b": 2, "c": 3},
            {"a": None, "b": 2, "c": 3},
            {"a": 1, "b": None, "c": 3},
            {"a": 1, "b": 2, "c": None},
            {"a": 1, "b": 2, "c": 3},
            {"a": 1, "b": 2, "c": 3},
        ]
        result = analyzer.analyze(data)
        # Uniform missing across columns -> MCAR
        assert result["missing_pattern"] in (PATTERN_MCAR, PATTERN_UNKNOWN)

    def test_mar_correlated_missing(self, analyzer):
        """Test MAR detected for correlated missing data."""
        # When col a is null, col b is also null > 50% of the time
        data = [
            {"a": None, "b": None, "c": 1},
            {"a": None, "b": None, "c": 2},
            {"a": None, "b": None, "c": 3},
            {"a": None, "b": None, "c": 4},
            {"a": None, "b": None, "c": 5},
            {"a": 1, "b": 2, "c": 6},
            {"a": 2, "b": 3, "c": 7},
            {"a": 3, "b": 4, "c": 8},
            {"a": 4, "b": 5, "c": 9},
            {"a": 5, "b": 6, "c": 10},
        ]
        result = analyzer.analyze(data)
        # Correlated nulls between a and b
        assert result["missing_pattern"] in (PATTERN_MAR, PATTERN_MNAR)

    def test_mnar_systematic_high_missing(self, analyzer):
        """Test MNAR detected for columns with > 30% missing."""
        data = [
            {"a": None, "b": 1},
            {"a": None, "b": 2},
            {"a": None, "b": 3},
            {"a": None, "b": 4},
            {"a": 1, "b": 5},
            {"a": 2, "b": 6},
            {"a": 3, "b": 7},
            {"a": 4, "b": 8},
            {"a": 5, "b": 9},
            {"a": 6, "b": 10},
        ]
        result = analyzer.analyze(data)
        # 40% missing in col a -> MNAR
        assert result["missing_pattern"] in (PATTERN_MNAR, PATTERN_MAR, PATTERN_UNKNOWN)

    def test_unknown_default(self, analyzer):
        """Test UNKNOWN returned when no clear pattern detected."""
        # Single column pattern detection
        values = [1, None, 3, 4, 5, 6, 7, 8, 9, 10]
        result = analyzer.detect_missing_pattern(values)
        assert result in (PATTERN_MCAR, PATTERN_MNAR, PATTERN_UNKNOWN)

    def test_empty_list(self, analyzer):
        """Test empty list returns UNKNOWN."""
        assert analyzer.detect_missing_pattern([]) == PATTERN_UNKNOWN

    def test_all_present(self, analyzer):
        """Test all present values return MCAR (no missing)."""
        result = analyzer.detect_missing_pattern([1, 2, 3, 4, 5])
        assert result == PATTERN_MCAR

    def test_all_null(self, analyzer):
        """Test all null returns MNAR."""
        result = analyzer.detect_missing_pattern([None, None, None])
        assert result == PATTERN_MNAR

    def test_insufficient_data(self, analyzer):
        """Test very small dataset returns a valid pattern."""
        result = analyzer.detect_missing_pattern([None])
        assert result in (PATTERN_MCAR, PATTERN_MAR, PATTERN_MNAR, PATTERN_UNKNOWN)


# ============================================================================
# TestFindRequiredGaps
# ============================================================================


class TestFindRequiredGaps:
    """Test required field gap finding."""

    def test_no_gaps(self, analyzer, complete_data):
        """Test no gaps when all required fields are present."""
        gaps = analyzer.find_required_gaps(complete_data, ["name", "age"])
        assert len(gaps) == 0

    def test_single_missing(self, analyzer):
        """Test single missing required field detected."""
        data = [
            {"name": "Alice", "age": 30},
            {"name": None, "age": 25},
        ]
        gaps = analyzer.find_required_gaps(data, ["name"])
        assert len(gaps) == 1
        assert gaps[0]["row_index"] == 1
        assert "name" in gaps[0]["missing_fields"]

    def test_multiple_missing(self, analyzer):
        """Test multiple missing fields in same row detected."""
        data = [{"name": None, "age": None, "email": "a@b.com"}]
        gaps = analyzer.find_required_gaps(data, ["name", "age", "email"])
        assert len(gaps) == 1
        assert len(gaps[0]["missing_fields"]) == 2

    def test_required_field_not_in_data(self, analyzer):
        """Test required field missing from data entirely."""
        data = [{"name": "Alice"}, {"name": "Bob"}]
        gaps = analyzer.find_required_gaps(data, ["email"])
        assert len(gaps) == 2  # email missing from both rows

    def test_all_required_present(self, analyzer, complete_data):
        """Test no gaps when all required fields present."""
        gaps = analyzer.find_required_gaps(complete_data, ["name"])
        assert len(gaps) == 0

    def test_empty_required_list(self, analyzer, complete_data):
        """Test empty required fields list produces no gaps."""
        gaps = analyzer.find_required_gaps(complete_data, [])
        assert len(gaps) == 0

    def test_empty_data(self, analyzer):
        """Test empty data produces no gaps."""
        gaps = analyzer.find_required_gaps([], ["name"])
        assert len(gaps) == 0

    def test_partial_gaps(self, analyzer):
        """Test partial gaps reported correctly."""
        data = [
            {"name": "Alice", "age": 30, "email": "a@b.com"},
            {"name": "Bob", "age": None, "email": "b@c.com"},
            {"name": "Charlie", "age": 35, "email": None},
        ]
        gaps = analyzer.find_required_gaps(data, ["age", "email"])
        assert len(gaps) == 2
        # Row 1: age missing
        assert gaps[0]["missing_fields"] == ["age"]
        # Row 2: email missing
        assert gaps[1]["missing_fields"] == ["email"]


# ============================================================================
# TestComputeRecordCompleteness
# ============================================================================


class TestComputeRecordCompleteness:
    """Test per-record completeness."""

    def test_fully_complete(self, analyzer):
        """Test fully complete record = 1.0."""
        result = analyzer.compute_record_completeness(
            {"a": 1, "b": 2, "c": 3}
        )
        assert result == pytest.approx(1.0, abs=0.01)

    def test_all_missing(self, analyzer):
        """Test all missing record = 0.0."""
        result = analyzer.compute_record_completeness(
            {"a": None, "b": None, "c": None}
        )
        assert result == pytest.approx(0.0, abs=0.01)

    def test_partial(self, analyzer):
        """Test partial record."""
        result = analyzer.compute_record_completeness(
            {"a": 1, "b": None, "c": 3}
        )
        assert result == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_empty_record(self, analyzer):
        """Test empty record = 1.0 (no expected fields)."""
        result = analyzer.compute_record_completeness({})
        assert result == pytest.approx(1.0, abs=0.01)

    def test_extra_fields_ignored(self, analyzer):
        """Test only expected fields are checked."""
        result = analyzer.compute_record_completeness(
            {"a": 1, "b": 2, "c": None, "d": 4},
            expected_fields=["a", "b"],
        )
        assert result == pytest.approx(1.0, abs=0.01)

    def test_expected_fields_subset(self, analyzer):
        """Test missing expected fields reduce completeness."""
        result = analyzer.compute_record_completeness(
            {"a": 1},
            expected_fields=["a", "b", "c"],
        )
        # b and c are missing from the record -> record.get returns None
        assert result == pytest.approx(1.0 / 3.0, abs=0.01)


# ============================================================================
# TestGetCoverageMatrix
# ============================================================================


class TestGetCoverageMatrix:
    """Test coverage matrix generation."""

    def test_all_filled(self, analyzer, complete_data):
        """Test all filled produces fill_rate 1.0 for all columns."""
        matrix = analyzer.get_coverage_matrix(complete_data)
        for col, rate in matrix.items():
            assert rate == pytest.approx(1.0, abs=0.01)

    def test_mixed_coverage(self, analyzer, partial_data):
        """Test mixed data produces varied fill rates."""
        matrix = analyzer.get_coverage_matrix(partial_data)
        assert matrix["name"] < 1.0  # has None and empty string
        assert 0.0 <= matrix["email"] <= 1.0

    def test_empty_data(self, analyzer):
        """Test empty data returns empty matrix."""
        result = analyzer.get_coverage_matrix([])
        assert result == {}

    def test_single_column(self, analyzer):
        """Test single column coverage."""
        data = [{"x": 1}, {"x": None}, {"x": 3}]
        matrix = analyzer.get_coverage_matrix(data)
        assert matrix["x"] == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_column_names_correct(self, analyzer, complete_data):
        """Test matrix keys match column names."""
        matrix = analyzer.get_coverage_matrix(complete_data)
        expected_cols = set(complete_data[0].keys())
        assert set(matrix.keys()) == expected_cols

    def test_values_between_0_and_1(self, analyzer, partial_data):
        """Test all fill_rate values are between 0 and 1."""
        matrix = analyzer.get_coverage_matrix(partial_data)
        for col, rate in matrix.items():
            assert 0.0 <= rate <= 1.0


# ============================================================================
# TestGenerateCompletenessIssues
# ============================================================================


class TestGenerateCompletenessIssues:
    """Test completeness issue generation."""

    def test_high_emptiness_issues(self, analyzer, all_null_data):
        """Test high missing rate generates issues."""
        issues = analyzer.generate_completeness_issues(all_null_data)
        assert len(issues) > 0

    def test_no_issues_for_complete_data(self, analyzer, complete_data):
        """Test no issues for fully complete data."""
        issues = analyzer.generate_completeness_issues(complete_data)
        assert len(issues) == 0

    def test_required_field_issues(self, analyzer, partial_data):
        """Test required field issues escalated in severity."""
        issues = analyzer.generate_completeness_issues(
            partial_data,
            required_fields=["name", "email"],
        )
        required_issues = [i for i in issues if i.get("details", {}).get("is_required")]
        assert len(required_issues) > 0

    def test_severity_levels(self, analyzer):
        """Test severity levels assigned correctly."""
        # 60% missing -> critical
        data = [{"x": None}] * 6 + [{"x": 1}] * 4
        issues = analyzer.generate_completeness_issues(data)
        severities = [i["severity"] for i in issues]
        assert SEVERITY_CRITICAL in severities or SEVERITY_HIGH in severities

    def test_descriptions(self, analyzer, partial_data):
        """Test issues contain descriptions."""
        issues = analyzer.generate_completeness_issues(partial_data)
        for issue in issues:
            assert "message" in issue
            assert len(issue["message"]) > 0

    def test_column_reference(self, analyzer, all_null_data):
        """Test issues reference the column."""
        issues = analyzer.generate_completeness_issues(all_null_data)
        column_issues = [i for i in issues if i.get("column") != "__dataset__"]
        for issue in column_issues:
            assert "column" in issue

    def test_empty_data(self, analyzer):
        """Test empty data produces no issues."""
        issues = analyzer.generate_completeness_issues([])
        assert len(issues) == 0

    def test_threshold_behavior(self, analyzer):
        """Test only columns above medium threshold generate issues."""
        # Less than 10% missing should only generate low severity
        data = [{"x": 1}] * 95 + [{"x": None}] * 5
        issues = analyzer.generate_completeness_issues(data)
        col_issues = [i for i in issues if i.get("column") == "x"]
        if col_issues:
            assert col_issues[0]["severity"] in (SEVERITY_LOW, SEVERITY_MEDIUM)


# ============================================================================
# TestStatistics
# ============================================================================


class TestStatistics:
    """Test aggregate statistics."""

    def test_initial(self, analyzer):
        """Test initial statistics are zeroed."""
        stats = analyzer.get_statistics()
        assert stats["analyses_completed"] == 0

    def test_post_analysis(self, analyzer, complete_data):
        """Test statistics updated after analysis."""
        analyzer.analyze(complete_data)
        stats = analyzer.get_statistics()
        assert stats["analyses_completed"] == 1
        assert stats["total_rows_analyzed"] == 5

    def test_accumulates(self, analyzer, complete_data):
        """Test statistics accumulate across analyses."""
        analyzer.analyze(complete_data)
        analyzer.analyze(complete_data)
        stats = analyzer.get_statistics()
        assert stats["analyses_completed"] == 2
        assert stats["total_rows_analyzed"] == 10

    def test_by_dimension(self, analyzer, partial_data):
        """Test missing cells tracked."""
        analyzer.analyze(partial_data)
        stats = analyzer.get_statistics()
        assert stats["total_missing_cells"] > 0


# ============================================================================
# TestProvenance
# ============================================================================


class TestProvenance:
    """Test provenance hash generation."""

    def test_sha256_length(self, analyzer, complete_data):
        """Test provenance hash is 64-char SHA-256."""
        result = analyzer.analyze(complete_data)
        assert len(result["provenance_hash"]) == 64

    def test_different_data_different_hashes(self, analyzer):
        """Test different data produces different hashes."""
        data_a = [{"x": 1}]
        data_b = [{"y": 2}]
        result_a = analyzer.analyze(data_a)
        result_b = analyzer.analyze(data_b)
        assert result_a["provenance_hash"] != result_b["provenance_hash"]

    def test_column_provenance(self, analyzer):
        """Test each column analysis has provenance hash."""
        data = [{"a": 1, "b": None}]
        result = analyzer.analyze(data)
        for col, col_result in result["column_completeness"].items():
            assert len(col_result["provenance_hash"]) == 64


# ============================================================================
# TestThreadSafety
# ============================================================================


class TestThreadSafety:
    """Test thread safety."""

    def test_concurrent_analysis(self, analyzer):
        """Test multiple threads can analyze concurrently."""
        errors = []
        data = [{"x": i, "y": i * 2} for i in range(100)]

        def do_analysis(name):
            try:
                analyzer.analyze(data)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=do_analysis, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = analyzer.get_statistics()
        assert stats["analyses_completed"] == 10

    def test_stats_consistency(self, analyzer):
        """Test stats remain consistent under concurrent access."""
        data = [{"x": i} for i in range(50)]

        def do_analysis(_):
            analyzer.analyze(data)

        threads = [threading.Thread(target=do_analysis, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = analyzer.get_statistics()
        assert stats["analyses_completed"] == 5
        assert stats["total_rows_analyzed"] == 250


# ============================================================================
# TestStorageAndRetrieval
# ============================================================================


class TestStorageAndRetrieval:
    """Test analysis storage, retrieval, and deletion."""

    def test_get_existing(self, analyzer, complete_data):
        """Test retrieving existing analysis."""
        result = analyzer.analyze(complete_data)
        retrieved = analyzer.get_analysis(result["analysis_id"])
        assert retrieved is not None
        assert retrieved["analysis_id"] == result["analysis_id"]

    def test_get_nonexistent(self, analyzer):
        """Test retrieving non-existent analysis returns None."""
        assert analyzer.get_analysis("CMP-doesnotexist") is None

    def test_list_analyses(self, analyzer, complete_data):
        """Test listing analyses."""
        analyzer.analyze(complete_data)
        analyzer.analyze(complete_data)
        analyses = analyzer.list_analyses()
        assert len(analyses) == 2

    def test_delete_existing(self, analyzer, complete_data):
        """Test deleting existing analysis."""
        result = analyzer.analyze(complete_data)
        assert analyzer.delete_analysis(result["analysis_id"]) is True
        assert analyzer.get_analysis(result["analysis_id"]) is None

    def test_delete_nonexistent(self, analyzer):
        """Test deleting non-existent analysis returns False."""
        assert analyzer.delete_analysis("CMP-doesnotexist") is False


# ============================================================================
# TestHelperFunction
# ============================================================================


class TestHelperFunction:
    """Test module-level helper functions."""

    def test_is_missing_none(self):
        """Test _is_missing for None."""
        assert _is_missing(None) is True

    def test_is_missing_empty_string(self):
        """Test _is_missing for empty string."""
        assert _is_missing("") is True

    def test_is_missing_whitespace(self):
        """Test _is_missing for whitespace-only string."""
        assert _is_missing("   ") is True

    def test_is_missing_tab(self):
        """Test _is_missing for tab string."""
        assert _is_missing("\t") is True

    def test_is_not_missing_value(self):
        """Test _is_missing for valid value."""
        assert _is_missing("hello") is False

    def test_is_not_missing_zero(self):
        """Test _is_missing for zero."""
        assert _is_missing(0) is False

    def test_is_not_missing_false(self):
        """Test _is_missing for False."""
        assert _is_missing(False) is False

    def test_safe_stdev_single(self):
        """Test _safe_stdev returns 0.0 for single value."""
        assert _safe_stdev([1.0]) == 0.0

    def test_safe_stdev_normal(self):
        """Test _safe_stdev computes correctly."""
        result = _safe_stdev([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result > 0


# ============================================================================
# TestSeverityClassification
# ============================================================================


class TestSeverityClassification:
    """Test severity classification logic."""

    def test_critical_severity(self, analyzer):
        """Test critical severity for high missing rate."""
        data = [{"x": None}] * 6 + [{"x": 1}] * 4
        result = analyzer.analyze(data)
        col_result = result["column_completeness"]["x"]
        assert col_result["severity"] == SEVERITY_CRITICAL

    def test_high_severity(self, analyzer):
        """Test high severity for moderate missing rate."""
        data = [{"x": None}] * 4 + [{"x": 1}] * 6
        result = analyzer.analyze(data)
        col_result = result["column_completeness"]["x"]
        assert col_result["severity"] == SEVERITY_HIGH

    def test_medium_severity(self, analyzer):
        """Test medium severity for low missing rate."""
        data = [{"x": None}] * 2 + [{"x": 1}] * 8
        result = analyzer.analyze(data)
        col_result = result["column_completeness"]["x"]
        assert col_result["severity"] == SEVERITY_MEDIUM

    def test_low_severity(self, analyzer):
        """Test low severity for minimal missing rate."""
        data = [{"x": None}] + [{"x": 1}] * 99
        result = analyzer.analyze(data)
        col_result = result["column_completeness"]["x"]
        assert col_result["severity"] == SEVERITY_LOW

    def test_info_severity(self, analyzer, complete_data):
        """Test info severity for zero missing rate."""
        result = analyzer.analyze(complete_data)
        for col, col_result in result["column_completeness"].items():
            assert col_result["severity"] == SEVERITY_INFO


# ============================================================================
# TestAnalysisId
# ============================================================================


class TestAnalysisId:
    """Test analysis ID generation."""

    def test_analysis_id_prefix(self, analyzer, complete_data):
        """Test analysis_id starts with CMP- prefix."""
        result = analyzer.analyze(complete_data)
        assert result["analysis_id"].startswith("CMP-")

    def test_unique_ids(self, analyzer, complete_data):
        """Test each analysis gets a unique ID."""
        result_a = analyzer.analyze(complete_data)
        result_b = analyzer.analyze(complete_data)
        assert result_a["analysis_id"] != result_b["analysis_id"]
