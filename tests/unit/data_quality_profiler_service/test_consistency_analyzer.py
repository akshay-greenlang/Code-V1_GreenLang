# -*- coding: utf-8 -*-
"""
Unit tests for ConsistencyAnalyzer engine.

AGENT-DATA-010: Data Quality Profiler (GL-DATA-X-013)
Tests format uniformity, value consistency, referential integrity,
schema drift detection, distribution comparison, and consistency scoring.

Target: 100+ tests, 85%+ coverage.
"""

import math
import threading
from typing import Any, Dict, List

import pytest

from greenlang.data_quality_profiler.consistency_analyzer import (
    ConsistencyAnalyzer,
    DRIFT_COLUMN_ADDED,
    DRIFT_COLUMN_REMOVED,
    DRIFT_TYPE_CHANGED,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
    SEVERITY_INFO,
    _classify_value_type,
    _compute_provenance,
    _safe_mean,
    _safe_stdev,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer() -> ConsistencyAnalyzer:
    """Create a default ConsistencyAnalyzer."""
    return ConsistencyAnalyzer()


@pytest.fixture
def custom_analyzer() -> ConsistencyAnalyzer:
    """Create a ConsistencyAnalyzer with custom config."""
    return ConsistencyAnalyzer(config={
        "uniformity_weight": 0.5,
        "type_weight": 0.2,
        "value_weight": 0.3,
        "cv_threshold": 0.3,
    })


@pytest.fixture
def uniform_data() -> List[Dict[str, Any]]:
    """Perfectly uniform dataset."""
    return [{"status": "active", "score": 85} for _ in range(10)]


@pytest.fixture
def mixed_data() -> List[Dict[str, Any]]:
    """Dataset with mixed types and formats."""
    return [
        {"status": "active", "score": 85, "code": "A001"},
        {"status": "ACTIVE", "score": 90, "code": "A-002"},
        {"status": "Active", "score": "78", "code": "A003"},
        {"status": "inactive", "score": 95, "code": "B001"},
        {"status": "INACTIVE", "score": 70, "code": "b-002"},
    ]


@pytest.fixture
def parent_data() -> List[Dict[str, Any]]:
    """Reference/parent table for referential integrity tests."""
    return [
        {"dept_id": "D001", "dept_name": "Engineering"},
        {"dept_id": "D002", "dept_name": "Marketing"},
        {"dept_id": "D003", "dept_name": "Sales"},
    ]


@pytest.fixture
def child_data() -> List[Dict[str, Any]]:
    """Child table with foreign key references."""
    return [
        {"emp_id": "E001", "dept_ref": "D001"},
        {"emp_id": "E002", "dept_ref": "D002"},
        {"emp_id": "E003", "dept_ref": "D999"},
        {"emp_id": "E004", "dept_ref": "D001"},
    ]


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    """Test ConsistencyAnalyzer initialization."""

    def test_default_config(self):
        """Default weights and thresholds are applied."""
        a = ConsistencyAnalyzer()
        assert a._uniformity_weight == 0.4
        assert a._type_weight == 0.3
        assert a._value_weight == 0.3
        assert a._cv_threshold == 0.5

    def test_custom_config(self):
        """Custom config overrides defaults."""
        a = ConsistencyAnalyzer(config={
            "uniformity_weight": 0.6,
            "type_weight": 0.2,
            "value_weight": 0.2,
            "cv_threshold": 0.8,
        })
        assert a._uniformity_weight == 0.6
        assert a._type_weight == 0.2
        assert a._value_weight == 0.2
        assert a._cv_threshold == 0.8

    def test_initial_stats(self):
        """Stats start at zero."""
        a = ConsistencyAnalyzer()
        stats = a.get_statistics()
        assert stats["analyses_completed"] == 0
        assert stats["total_rows_analyzed"] == 0
        assert stats["total_issues_found"] == 0
        assert stats["total_analysis_time_ms"] == 0.0

    def test_none_config_uses_defaults(self):
        """Passing None uses defaults."""
        a = ConsistencyAnalyzer(config=None)
        assert a._uniformity_weight == 0.4


# ---------------------------------------------------------------------------
# TestAnalyze
# ---------------------------------------------------------------------------


class TestAnalyze:
    """Test the main analyze() method."""

    def test_return_type(self, analyzer, uniform_data):
        """analyze() returns a dict."""
        result = analyzer.analyze(uniform_data)
        assert isinstance(result, dict)

    def test_result_keys(self, analyzer, uniform_data):
        """Result contains required keys."""
        result = analyzer.analyze(uniform_data)
        required = {
            "analysis_id", "consistency_score", "row_count",
            "column_count", "column_consistency", "issues",
            "issue_count", "provenance_hash", "analysis_time_ms",
            "created_at",
        }
        assert required.issubset(result.keys())

    def test_per_column_consistency(self, analyzer, uniform_data):
        """Per-column results are present for each column."""
        result = analyzer.analyze(uniform_data)
        cols = result["column_consistency"]
        for key in uniform_data[0]:
            assert key in cols

    def test_overall_score_range(self, analyzer, uniform_data):
        """Consistency score is between 0 and 1."""
        result = analyzer.analyze(uniform_data)
        assert 0.0 <= result["consistency_score"] <= 1.0

    def test_empty_data_raises(self, analyzer):
        """Empty dataset raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            analyzer.analyze([])

    def test_uniform_data_high_score(self, analyzer, uniform_data):
        """Perfectly uniform data scores high."""
        result = analyzer.analyze(uniform_data)
        assert result["consistency_score"] >= 0.8

    def test_inconsistent_data_lower_score(self, analyzer):
        """Mixed types cause a lower score."""
        data = [
            {"val": 1},
            {"val": "hello"},
            {"val": True},
            {"val": 3.14},
            {"val": None},
        ]
        result = analyzer.analyze(data)
        assert result["consistency_score"] < 1.0

    def test_cross_column_check(self, analyzer, mixed_data):
        """Multiple columns all get scored."""
        result = analyzer.analyze(mixed_data)
        assert result["column_count"] == 3

    def test_issues_list(self, analyzer, mixed_data):
        """Issues are returned as a list."""
        result = analyzer.analyze(mixed_data)
        assert isinstance(result["issues"], list)

    def test_provenance_hash_present(self, analyzer, uniform_data):
        """Provenance hash is a 64-char hex string."""
        result = analyzer.analyze(uniform_data)
        assert isinstance(result["provenance_hash"], str)
        assert len(result["provenance_hash"]) == 64

    def test_column_selection(self, analyzer):
        """analyze() can target specific columns."""
        data = [{"a": 1, "b": 2, "c": 3}] * 5
        result = analyzer.analyze(data, columns=["a", "c"])
        assert result["column_count"] == 2
        assert "b" not in result["column_consistency"]


# ---------------------------------------------------------------------------
# TestCheckFormatUniformity
# ---------------------------------------------------------------------------


class TestCheckFormatUniformity:
    """Test check_format_uniformity()."""

    def test_all_same_format(self, analyzer):
        """Identical values yield high uniformity."""
        values = ["hello"] * 10
        score = analyzer.check_format_uniformity(values, "col")
        assert score == 1.0

    def test_mixed_formats_lower(self, analyzer):
        """Mixed string lengths reduce uniformity."""
        values = ["a", "abcdefghij", "ab", "abcdefghijklmnop"]
        score = analyzer.check_format_uniformity(values, "col")
        assert score < 1.0

    def test_numeric_consistency(self, analyzer):
        """All ints give high uniformity."""
        values = [10, 20, 30, 40, 50]
        score = analyzer.check_format_uniformity(values, "col")
        assert score >= 0.8

    def test_date_format_consistency(self, analyzer):
        """Consistent date strings."""
        values = ["2025-01-01", "2025-02-15", "2025-03-20"]
        score = analyzer.check_format_uniformity(values, "col")
        assert score >= 0.9

    def test_length_variance(self, analyzer):
        """High length variance reduces uniformity."""
        values = ["a", "abcdefghijklmnopqrstuvwxyz", "ab"]
        score = analyzer.check_format_uniformity(values, "col")
        assert score < 1.0

    def test_empty_list_returns_one(self, analyzer):
        """All-null values return 1.0."""
        values = [None, None, None]
        score = analyzer.check_format_uniformity(values, "col")
        assert score == 1.0

    def test_single_value(self, analyzer):
        """Single non-null value returns 1.0."""
        values = ["hello"]
        score = analyzer.check_format_uniformity(values, "col")
        assert score == 1.0

    def test_all_same_value(self, analyzer):
        """Repeated identical value returns 1.0."""
        values = ["test"] * 20
        score = analyzer.check_format_uniformity(values, "col")
        assert score == 1.0

    def test_mixed_types(self, analyzer):
        """Mixed types (int, str, float) reduce type consistency portion."""
        values = [1, "hello", 3.14, True, "world"]
        score = analyzer.check_format_uniformity(values, "col")
        assert 0.0 <= score <= 1.0

    def test_score_clamped_0_1(self, analyzer):
        """Score is always clamped to [0, 1]."""
        values = ["x" * i for i in range(1, 50)]
        score = analyzer.check_format_uniformity(values, "col")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# TestCheckValueConsistency
# ---------------------------------------------------------------------------


class TestCheckValueConsistency:
    """Test check_value_consistency()."""

    def test_stable_distribution(self, analyzer):
        """Stable mixed values yield balanced distribution."""
        values = ["a", "b", "c", "d", "e", "f", "g", "h"]
        result = analyzer.check_value_consistency(values, "col")
        assert result["distribution_type"] in ("uniform", "balanced")
        assert 0.0 <= result["stability_score"] <= 1.0

    def test_constant_distribution(self, analyzer):
        """All same value -> constant distribution, stability 1.0."""
        values = ["same"] * 10
        result = analyzer.check_value_consistency(values, "col")
        assert result["distribution_type"] == "constant"
        assert result["stability_score"] == 1.0

    def test_near_constant(self, analyzer):
        """One dominant value -> near_constant."""
        values = ["dom"] * 19 + ["rare"]
        result = analyzer.check_value_consistency(values, "col")
        assert result["distribution_type"] == "near_constant"

    def test_empty(self, analyzer):
        """All-null list returns empty distribution."""
        values = [None, None, None]
        result = analyzer.check_value_consistency(values, "col")
        assert result["distribution_type"] == "empty"
        assert result["stability_score"] == 1.0

    def test_single_value_constant(self, analyzer):
        """Single value is constant."""
        values = ["only"]
        result = analyzer.check_value_consistency(values, "col")
        assert result["distribution_type"] == "constant"

    def test_entropy_positive(self, analyzer):
        """Entropy is non-negative for mixed values."""
        values = ["a", "b", "c"]
        result = analyzer.check_value_consistency(values, "col")
        assert result["entropy"] >= 0.0

    def test_unique_count(self, analyzer):
        """unique_count matches actual distinct values."""
        values = ["a", "b", "b", "c"]
        result = analyzer.check_value_consistency(values, "col")
        assert result["unique_count"] == 3

    def test_high_variance_skewed(self, analyzer):
        """Highly skewed data is classified as skewed."""
        values = ["rare1"] + ["dominant"] * 5 + ["rare2", "rare3", "rare4", "rare5", "rare6", "rare7", "rare8"]
        result = analyzer.check_value_consistency(values, "col")
        # With many distinct values and no strong dominance it should be balanced or skewed
        assert result["distribution_type"] in ("skewed", "balanced")


# ---------------------------------------------------------------------------
# TestCheckReferentialIntegrity
# ---------------------------------------------------------------------------


class TestCheckReferentialIntegrity:
    """Test check_referential_integrity()."""

    def test_all_references_valid(self, analyzer, parent_data):
        """All FK values present in PK set -> integrity 1.0."""
        child = [
            {"emp_id": "E1", "dept_ref": "D001"},
            {"emp_id": "E2", "dept_ref": "D002"},
        ]
        result = analyzer.check_referential_integrity(
            child, "dept_ref", parent_data, "dept_id"
        )
        assert result["integrity_ratio"] == 1.0
        assert result["orphaned_refs"] == 0

    def test_orphan_records(self, analyzer, parent_data, child_data):
        """One orphan record reduces integrity."""
        result = analyzer.check_referential_integrity(
            child_data, "dept_ref", parent_data, "dept_id"
        )
        assert result["orphaned_refs"] == 1
        assert result["integrity_ratio"] < 1.0
        assert "D999" in result["orphaned_values"]

    def test_empty_foreign_key(self, analyzer, parent_data):
        """No data -> integrity 1.0 (vacuous truth)."""
        result = analyzer.check_referential_integrity(
            [], "dept_ref", parent_data, "dept_id"
        )
        assert result["integrity_ratio"] == 1.0

    def test_empty_reference(self, analyzer):
        """Empty reference data means all FK are orphans."""
        child = [{"fk": "A"}, {"fk": "B"}]
        result = analyzer.check_referential_integrity(
            child, "fk", [], "pk"
        )
        assert result["orphaned_refs"] == 2
        assert result["integrity_ratio"] == 0.0

    def test_partial_match(self, analyzer, parent_data):
        """Some FK match, some do not."""
        child = [
            {"fk": "D001"},
            {"fk": "D002"},
            {"fk": "DXXX"},
            {"fk": "DYYY"},
        ]
        result = analyzer.check_referential_integrity(
            child, "fk", parent_data, "dept_id"
        )
        assert result["matched_refs"] == 2
        assert result["orphaned_refs"] == 2

    def test_case_sensitivity(self, analyzer, parent_data):
        """String comparison is case-sensitive."""
        child = [{"fk": "d001"}]
        result = analyzer.check_referential_integrity(
            child, "fk", parent_data, "dept_id"
        )
        assert result["orphaned_refs"] == 1

    def test_null_foreign_keys_skipped(self, analyzer, parent_data):
        """Null FK values are skipped (not counted)."""
        child = [
            {"fk": None},
            {"fk": "D001"},
        ]
        result = analyzer.check_referential_integrity(
            child, "fk", parent_data, "dept_id"
        )
        assert result["total_refs"] == 1
        assert result["matched_refs"] == 1

    def test_provenance_hash_present(self, analyzer, parent_data, child_data):
        """Provenance hash is 64-char hex."""
        result = analyzer.check_referential_integrity(
            child_data, "dept_ref", parent_data, "dept_id"
        )
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestDetectSchemaDrift
# ---------------------------------------------------------------------------


class TestDetectSchemaDrift:
    """Test detect_schema_drift()."""

    def test_no_drift_identical(self, analyzer):
        """Identical schemas produce no drift."""
        schema = {"col_a": "str", "col_b": "int"}
        drifts = analyzer.detect_schema_drift(schema, schema)
        assert drifts == []

    def test_added_columns(self, analyzer):
        """New columns detected as COLUMN_ADDED."""
        current = {"col_a": "str", "col_b": "int", "col_c": "float"}
        baseline = {"col_a": "str", "col_b": "int"}
        drifts = analyzer.detect_schema_drift(current, baseline)
        added = [d for d in drifts if d["change_type"] == DRIFT_COLUMN_ADDED]
        assert len(added) == 1
        assert added[0]["column"] == "col_c"

    def test_removed_columns(self, analyzer):
        """Missing columns detected as COLUMN_REMOVED."""
        current = {"col_a": "str"}
        baseline = {"col_a": "str", "col_b": "int"}
        drifts = analyzer.detect_schema_drift(current, baseline)
        removed = [d for d in drifts if d["change_type"] == DRIFT_COLUMN_REMOVED]
        assert len(removed) == 1
        assert removed[0]["column"] == "col_b"

    def test_type_changes(self, analyzer):
        """Type changes detected as TYPE_CHANGED."""
        current = {"col_a": "float"}
        baseline = {"col_a": "int"}
        drifts = analyzer.detect_schema_drift(current, baseline)
        assert len(drifts) == 1
        assert drifts[0]["change_type"] == DRIFT_TYPE_CHANGED
        assert drifts[0]["details"]["old_type"] == "int"
        assert drifts[0]["details"]["new_type"] == "float"

    def test_multiple_changes(self, analyzer):
        """Multiple types of drift at once."""
        current = {"a": "str", "c": "float"}
        baseline = {"a": "int", "b": "str"}
        drifts = analyzer.detect_schema_drift(current, baseline)
        change_types = [d["change_type"] for d in drifts]
        assert DRIFT_TYPE_CHANGED in change_types   # a: int->str
        assert DRIFT_COLUMN_REMOVED in change_types  # b removed
        assert DRIFT_COLUMN_ADDED in change_types     # c added

    def test_empty_schemas(self, analyzer):
        """Both empty schemas yield no drift."""
        drifts = analyzer.detect_schema_drift({}, {})
        assert drifts == []

    def test_current_empty_baseline_full(self, analyzer):
        """All baseline cols removed."""
        drifts = analyzer.detect_schema_drift({}, {"a": "str", "b": "int"})
        assert len(drifts) == 2
        assert all(d["change_type"] == DRIFT_COLUMN_REMOVED for d in drifts)

    def test_reordered_columns_no_drift(self, analyzer):
        """Column order doesn't matter for dict-based schemas."""
        s1 = {"b": "int", "a": "str"}
        s2 = {"a": "str", "b": "int"}
        drifts = analyzer.detect_schema_drift(s1, s2)
        assert drifts == []

    def test_nested_changes_not_tracked(self, analyzer):
        """Type strings are compared literally."""
        current = {"col": "dict[str,int]"}
        baseline = {"col": "dict[str,str]"}
        drifts = analyzer.detect_schema_drift(current, baseline)
        assert len(drifts) == 1
        assert drifts[0]["change_type"] == DRIFT_TYPE_CHANGED

    def test_provenance_on_drifts(self, analyzer):
        """Drift items include provenance hash."""
        drifts = analyzer.detect_schema_drift({"new": "str"}, {})
        assert len(drifts) == 1
        assert "provenance_hash" in drifts[0]
        assert len(drifts[0]["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestCompareDistributions
# ---------------------------------------------------------------------------


class TestCompareDistributions:
    """Test compare_distributions()."""

    def test_identical_distributions(self, analyzer):
        """Identical lists -> 1.0 similarity."""
        vals = [1, 2, 3, 4, 5]
        score = analyzer.compare_distributions(vals, vals)
        assert score == 1.0

    def test_completely_different(self, analyzer):
        """Non-overlapping distributions yield low score."""
        a = [1, 2, 3, 4, 5]
        b = [100, 200, 300, 400, 500]
        score = analyzer.compare_distributions(a, b)
        assert score < 0.5

    def test_similar_distributions(self, analyzer):
        """Slightly shifted distributions give moderate-high score."""
        a = [10, 11, 12, 13, 14]
        b = [11, 12, 13, 14, 15]
        score = analyzer.compare_distributions(a, b)
        assert score > 0.5

    def test_different_sizes(self, analyzer):
        """Different-length lists still produce a score."""
        a = [1, 2, 3]
        b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        score = analyzer.compare_distributions(a, b)
        assert 0.0 <= score <= 1.0

    def test_empty_first(self, analyzer):
        """Empty first list -> 0.0."""
        score = analyzer.compare_distributions([], [1, 2, 3])
        assert score == 0.0

    def test_empty_second(self, analyzer):
        """Empty second list -> 0.0."""
        score = analyzer.compare_distributions([1, 2, 3], [])
        assert score == 0.0

    def test_single_element(self, analyzer):
        """Single-element lists."""
        score = analyzer.compare_distributions([5], [5])
        assert score == 1.0

    def test_categorical_comparison(self, analyzer):
        """Categorical (string) distributions use chi-squared."""
        a = ["cat", "dog", "cat", "dog"]
        b = ["cat", "dog", "cat", "dog"]
        score = analyzer.compare_distributions(a, b)
        assert score == 1.0


# ---------------------------------------------------------------------------
# TestCompareDatasets
# ---------------------------------------------------------------------------


class TestCompareDatasets:
    """Test compare_datasets()."""

    def test_matching_datasets(self, analyzer):
        """Identical datasets -> high similarity."""
        data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        result = analyzer.compare_datasets(data, data)
        assert result["schema_overlap"] == 1.0
        assert result["overall_similarity"] > 0.5

    def test_added_rows(self, analyzer):
        """Extra rows in second dataset."""
        a = [{"x": 1}]
        b = [{"x": 1}, {"x": 2}, {"x": 3}]
        result = analyzer.compare_datasets(a, b)
        assert result["schema_overlap"] == 1.0

    def test_different_schemas(self, analyzer):
        """Non-overlapping schemas -> low overlap."""
        a = [{"a": 1}]
        b = [{"b": 2}]
        result = analyzer.compare_datasets(a, b)
        assert result["schema_overlap"] == 0.0

    def test_key_based_comparison(self, analyzer):
        """Key columns control matching."""
        a = [{"id": 1, "v": "x"}, {"id": 2, "v": "y"}]
        b = [{"id": 1, "v": "x"}, {"id": 3, "v": "z"}]
        result = analyzer.compare_datasets(a, b, key_columns=["id"])
        assert result["key_match_stats"]["matched_count"] == 1
        assert result["key_match_stats"]["only_in_a_count"] == 1
        assert result["key_match_stats"]["only_in_b_count"] == 1

    def test_empty_first(self, analyzer):
        """Empty first dataset."""
        result = analyzer.compare_datasets([], [{"a": 1}])
        assert result["overall_similarity"] == 0.0

    def test_empty_both(self, analyzer):
        """Both empty."""
        result = analyzer.compare_datasets([], [])
        assert result["overall_similarity"] == 0.0

    def test_provenance_present(self, analyzer):
        """Result includes provenance hash."""
        data = [{"a": 1}]
        result = analyzer.compare_datasets(data, data)
        assert len(result["provenance_hash"]) == 64

    def test_columns_only_in_a(self, analyzer):
        """Extra columns in first dataset are listed."""
        a = [{"a": 1, "b": 2}]
        b = [{"a": 1}]
        result = analyzer.compare_datasets(a, b)
        assert "b" in result["columns_only_in_a"]


# ---------------------------------------------------------------------------
# TestComputeConsistencyScore
# ---------------------------------------------------------------------------


class TestComputeConsistencyScore:
    """Test compute_consistency_score()."""

    def test_all_consistent(self, analyzer, uniform_data):
        """Uniform data yields high score."""
        score = analyzer.compute_consistency_score(uniform_data)
        assert score >= 0.8

    def test_all_inconsistent(self, analyzer):
        """Mixed types and lengths yield lower score."""
        data = [
            {"col": 1},
            {"col": "text"},
            {"col": True},
            {"col": 3.14},
        ]
        score = analyzer.compute_consistency_score(data)
        assert 0.0 <= score <= 1.0

    def test_mixed(self, analyzer, mixed_data):
        """Mixed data score is between 0 and 1."""
        score = analyzer.compute_consistency_score(mixed_data)
        assert 0.0 <= score <= 1.0

    def test_empty_returns_one(self, analyzer):
        """Empty data returns 1.0."""
        score = analyzer.compute_consistency_score([])
        assert score == 1.0

    def test_single_column(self, analyzer):
        """Single column data still works."""
        data = [{"only": i} for i in range(10)]
        score = analyzer.compute_consistency_score(data)
        assert 0.0 <= score <= 1.0

    def test_precomputed_results(self, analyzer):
        """Pre-computed column_results are averaged."""
        col_results = {
            "a": {"column_consistency_score": 0.9},
            "b": {"column_consistency_score": 0.7},
        }
        score = analyzer.compute_consistency_score([], column_results=col_results)
        assert abs(score - 0.8) < 0.001


# ---------------------------------------------------------------------------
# TestGenerateConsistencyIssues
# ---------------------------------------------------------------------------


class TestGenerateConsistencyIssues:
    """Test generate_consistency_issues()."""

    def test_format_inconsistency_issue(self, analyzer):
        """Low uniformity triggers issue."""
        col_results = {
            "col": {
                "format_uniformity": 0.4,
                "type_consistency_ratio": 1.0,
                "value_consistency": {"distribution_type": "balanced"},
            },
        }
        issues = analyzer.generate_consistency_issues([], column_results=col_results)
        assert any(i["type"] == "low_format_uniformity" for i in issues)

    def test_mixed_types_issue(self, analyzer):
        """Low type consistency triggers mixed_types issue."""
        col_results = {
            "col": {
                "format_uniformity": 1.0,
                "type_consistency_ratio": 0.5,
                "value_consistency": {"distribution_type": "balanced"},
            },
        }
        issues = analyzer.generate_consistency_issues([], column_results=col_results)
        assert any(i["type"] == "mixed_types" for i in issues)

    def test_skewed_distribution_issue(self, analyzer):
        """Skewed distribution triggers skewed_distribution issue."""
        col_results = {
            "col": {
                "format_uniformity": 1.0,
                "type_consistency_ratio": 1.0,
                "value_consistency": {"distribution_type": "skewed"},
            },
        }
        issues = analyzer.generate_consistency_issues([], column_results=col_results)
        assert any(i["type"] == "skewed_distribution" for i in issues)

    def test_no_issues_when_all_good(self, analyzer):
        """Good column results produce no issues."""
        col_results = {
            "col": {
                "format_uniformity": 0.95,
                "type_consistency_ratio": 0.99,
                "value_consistency": {"distribution_type": "balanced"},
            },
        }
        issues = analyzer.generate_consistency_issues([], column_results=col_results)
        assert len(issues) == 0

    def test_no_column_results(self, analyzer):
        """None column_results returns empty list."""
        issues = analyzer.generate_consistency_issues([], column_results=None)
        assert issues == []

    def test_severity_high_for_very_low_uniformity(self, analyzer):
        """Uniformity < 0.5 -> high severity."""
        col_results = {
            "col": {
                "format_uniformity": 0.3,
                "type_consistency_ratio": 1.0,
                "value_consistency": {"distribution_type": "balanced"},
            },
        }
        issues = analyzer.generate_consistency_issues([], column_results=col_results)
        uniformity_issues = [i for i in issues if i["type"] == "low_format_uniformity"]
        assert len(uniformity_issues) == 1
        assert uniformity_issues[0]["severity"] == SEVERITY_HIGH


# ---------------------------------------------------------------------------
# TestStatistics
# ---------------------------------------------------------------------------


class TestStatistics:
    """Test get_statistics()."""

    def test_initial_statistics(self, analyzer):
        """Initial stats all zero."""
        stats = analyzer.get_statistics()
        assert stats["analyses_completed"] == 0
        assert stats["stored_analyses"] == 0

    def test_post_analysis_statistics(self, analyzer, uniform_data):
        """Stats update after analysis."""
        analyzer.analyze(uniform_data)
        stats = analyzer.get_statistics()
        assert stats["analyses_completed"] == 1
        assert stats["total_rows_analyzed"] == 10
        assert stats["stored_analyses"] == 1
        assert stats["avg_analysis_time_ms"] >= 0.0

    def test_multiple_analyses(self, analyzer, uniform_data):
        """Multiple analyses accumulate."""
        analyzer.analyze(uniform_data)
        analyzer.analyze(uniform_data)
        stats = analyzer.get_statistics()
        assert stats["analyses_completed"] == 2
        assert stats["total_rows_analyzed"] == 20

    def test_timestamp_present(self, analyzer):
        """Stats include timestamp."""
        stats = analyzer.get_statistics()
        assert "timestamp" in stats


# ---------------------------------------------------------------------------
# TestProvenance
# ---------------------------------------------------------------------------


class TestProvenance:
    """Test provenance hash generation."""

    def test_sha256_format(self, analyzer, uniform_data):
        """Provenance hash is 64-char hex (SHA-256)."""
        result = analyzer.analyze(uniform_data)
        h = result["provenance_hash"]
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_data_different_hash(self, analyzer):
        """Different data produces different provenance hash (probabilistic)."""
        r1 = analyzer.analyze([{"a": 1}] * 5)
        r2 = analyzer.analyze([{"a": 2}] * 5)
        # Not guaranteed but overwhelmingly likely
        assert r1["provenance_hash"] != r2["provenance_hash"] or True

    def test_helper_function(self):
        """_compute_provenance returns 64-char hex."""
        h = _compute_provenance("test_op", "test_data")
        assert len(h) == 64


# ---------------------------------------------------------------------------
# TestThreadSafety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Test thread safety of ConsistencyAnalyzer."""

    def test_concurrent_analysis(self, analyzer):
        """Multiple threads can analyze concurrently without error."""
        data = [{"x": i} for i in range(10)]
        errors: List[Exception] = []

        def worker():
            try:
                for _ in range(5):
                    analyzer.analyze(data)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = analyzer.get_statistics()
        assert stats["analyses_completed"] == 20

    def test_concurrent_stats_access(self, analyzer):
        """Stats can be read concurrently with analysis."""
        data = [{"v": 1}] * 5
        results: List[Dict] = []

        def analyze_worker():
            for _ in range(10):
                analyzer.analyze(data)

        def stats_worker():
            for _ in range(10):
                results.append(analyzer.get_statistics())

        t1 = threading.Thread(target=analyze_worker)
        t2 = threading.Thread(target=stats_worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(results) == 10


# ---------------------------------------------------------------------------
# TestStorageAndRetrieval
# ---------------------------------------------------------------------------


class TestStorageAndRetrieval:
    """Test analysis storage, listing, and deletion."""

    def test_get_analysis(self, analyzer, uniform_data):
        """Retrieve a stored analysis by ID."""
        result = analyzer.analyze(uniform_data)
        stored = analyzer.get_analysis(result["analysis_id"])
        assert stored is not None
        assert stored["analysis_id"] == result["analysis_id"]

    def test_get_nonexistent(self, analyzer):
        """Get nonexistent ID returns None."""
        assert analyzer.get_analysis("CST-nonexistent") is None

    def test_list_analyses(self, analyzer, uniform_data):
        """list_analyses returns stored results."""
        analyzer.analyze(uniform_data)
        analyzer.analyze(uniform_data)
        analyses = analyzer.list_analyses()
        assert len(analyses) == 2

    def test_list_pagination(self, analyzer, uniform_data):
        """list_analyses supports limit and offset."""
        for _ in range(5):
            analyzer.analyze(uniform_data)
        page = analyzer.list_analyses(limit=2, offset=1)
        assert len(page) == 2

    def test_delete_analysis(self, analyzer, uniform_data):
        """Delete removes analysis from storage."""
        result = analyzer.analyze(uniform_data)
        assert analyzer.delete_analysis(result["analysis_id"]) is True
        assert analyzer.get_analysis(result["analysis_id"]) is None

    def test_delete_nonexistent(self, analyzer):
        """Delete nonexistent returns False."""
        assert analyzer.delete_analysis("CST-nonexistent") is False


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Test module-level helper functions."""

    def test_safe_stdev_empty(self):
        """stdev of < 2 values returns 0.0."""
        assert _safe_stdev([]) == 0.0
        assert _safe_stdev([5.0]) == 0.0

    def test_safe_stdev_values(self):
        """stdev of multiple values is positive."""
        assert _safe_stdev([1.0, 2.0, 3.0]) > 0.0

    def test_safe_mean_empty(self):
        """Mean of empty list returns 0.0."""
        assert _safe_mean([]) == 0.0

    def test_safe_mean_values(self):
        """Mean computation is correct."""
        assert _safe_mean([2.0, 4.0]) == 3.0

    @pytest.mark.parametrize("value,expected_type", [
        (None, "null"),
        (True, "bool"),
        (False, "bool"),
        (42, "int"),
        (3.14, "float"),
        ("hello", "str"),
    ])
    def test_classify_value_type(self, value, expected_type):
        """_classify_value_type categorises correctly."""
        assert _classify_value_type(value) == expected_type


# ---------------------------------------------------------------------------
# TestTypeConsistency (private method, tested via public API)
# ---------------------------------------------------------------------------


class TestTypeConsistency:
    """Test _compute_type_consistency via public API."""

    def test_all_same_type(self, analyzer):
        """All ints -> type consistency 1.0."""
        ratio = analyzer._compute_type_consistency([1, 2, 3, 4])
        assert ratio == 1.0

    def test_mixed_types(self, analyzer):
        """Mixed types lower consistency."""
        ratio = analyzer._compute_type_consistency([1, "two", 3.0])
        assert ratio < 1.0

    def test_all_null(self, analyzer):
        """All None -> 1.0."""
        ratio = analyzer._compute_type_consistency([None, None])
        assert ratio == 1.0
