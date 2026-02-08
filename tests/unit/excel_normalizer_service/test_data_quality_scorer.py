# -*- coding: utf-8 -*-
"""
Unit Tests for DataQualityScorer (AGENT-DATA-002)

Tests data quality scoring including completeness, accuracy, consistency,
outlier detection, duplicate detection, and quality level computation.

Coverage target: 85%+ of data_quality_scorer.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline DataQualityScorer
# ---------------------------------------------------------------------------


class DataQualityScorer:
    """Scores data quality along completeness, accuracy, and consistency."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._completeness_weight: float = self._config.get("completeness_weight", 0.4)
        self._accuracy_weight: float = self._config.get("accuracy_weight", 0.35)
        self._consistency_weight: float = self._config.get("consistency_weight", 0.25)
        self._stats: Dict[str, int] = {"files_scored": 0, "rows_scored": 0, "issues_found": 0}

    def score_file(self, headers: List[str], rows: List[List[Any]]) -> Dict[str, Any]:
        completeness = self.score_completeness(headers, rows)
        accuracy = self.score_accuracy(headers, rows)
        consistency = self.score_consistency(headers, rows)
        overall = (completeness * self._completeness_weight +
                   accuracy * self._accuracy_weight +
                   consistency * self._consistency_weight)
        level = self.compute_quality_level(overall)
        issues = self.generate_issues(headers, rows)
        self._stats["files_scored"] += 1
        self._stats["rows_scored"] += len(rows)
        self._stats["issues_found"] += len(issues)
        return {
            "overall_score": round(overall, 4), "completeness_score": round(completeness, 4),
            "accuracy_score": round(accuracy, 4), "consistency_score": round(consistency, 4),
            "quality_level": level, "total_rows": len(rows),
            "issues": issues, "column_scores": self.score_columns(headers, rows),
        }

    def score_completeness(self, headers: List[str], rows: List[List[Any]]) -> float:
        if not rows or not headers:
            return 0.0
        total_cells = len(rows) * len(headers)
        filled = 0
        for row in rows:
            for idx in range(min(len(row), len(headers))):
                if row[idx] is not None and str(row[idx]).strip() != "":
                    filled += 1
        return filled / total_cells if total_cells > 0 else 0.0

    def score_accuracy(self, headers: List[str], rows: List[List[Any]]) -> float:
        if not rows:
            return 0.0
        # Simplified: ratio of non-error values (no type mismatches detected)
        total, valid = 0, 0
        for row in rows:
            for val in row:
                total += 1
                if val is not None:
                    valid += 1
        return valid / total if total > 0 else 0.0

    def score_consistency(self, headers: List[str], rows: List[List[Any]]) -> float:
        if len(rows) < 2:
            return 1.0
        # Check column type consistency across rows
        col_count = len(headers) if headers else (len(rows[0]) if rows else 0)
        consistent_cols = 0
        for col_idx in range(col_count):
            types = set()
            for row in rows:
                if col_idx < len(row) and row[col_idx] is not None:
                    types.add(type(row[col_idx]).__name__)
            if len(types) <= 1:
                consistent_cols += 1
        return consistent_cols / col_count if col_count > 0 else 1.0

    def score_column(self, values: List[Any]) -> float:
        if not values:
            return 0.0
        filled = sum(1 for v in values if v is not None and str(v).strip() != "")
        return filled / len(values)

    def score_columns(self, headers: List[str], rows: List[List[Any]]) -> Dict[str, float]:
        result = {}
        for col_idx, header in enumerate(headers):
            values = [row[col_idx] for row in rows if col_idx < len(row)]
            result[header] = round(self.score_column(values), 4)
        return result

    def detect_outliers(self, values: List[float], method: str = "iqr") -> List[int]:
        clean = [(i, v) for i, v in enumerate(values) if v is not None]
        if len(clean) < 4:
            return []
        nums = [v for _, v in clean]
        if method == "iqr":
            q1 = sorted(nums)[len(nums) // 4]
            q3 = sorted(nums)[3 * len(nums) // 4]
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            return [i for i, v in clean if v < low or v > high]
        elif method == "zscore":
            mean = statistics.mean(nums)
            stdev = statistics.stdev(nums) if len(nums) > 1 else 0
            if stdev == 0:
                return []
            return [i for i, v in clean if abs((v - mean) / stdev) > 3]
        return []

    def detect_duplicates(self, rows: List[List[Any]], method: str = "exact",
                          key_columns: Optional[List[int]] = None) -> List[int]:
        seen = {}
        duplicates = []
        for idx, row in enumerate(rows):
            if method == "exact":
                key = tuple(str(v) for v in row)
            elif method == "key" and key_columns:
                key = tuple(str(row[c]) for c in key_columns if c < len(row))
            else:
                key = tuple(str(v) for v in row)
            if key in seen:
                duplicates.append(idx)
            else:
                seen[key] = idx
        return duplicates

    def compute_quality_level(self, score: float) -> str:
        if score >= 0.9:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.5:
            return "fair"
        return "poor"

    def generate_issues(self, headers: List[str], rows: List[List[Any]]) -> List[Dict[str, Any]]:
        issues = []
        for col_idx, header in enumerate(headers):
            null_count = sum(1 for row in rows if col_idx >= len(row) or row[col_idx] is None or str(row[col_idx]).strip() == "")
            if null_count > 0:
                issues.append({"type": "null_values", "column": header, "count": null_count,
                               "severity": "warning" if null_count < len(rows) / 2 else "error"})
        duplicates = self.detect_duplicates(rows)
        if duplicates:
            issues.append({"type": "duplicates", "count": len(duplicates), "severity": "warning"})
        return issues

    def get_statistics(self) -> Dict[str, Any]:
        return dict(self._stats)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestDataQualityScorerInit:
    def test_default_creation(self):
        s = DataQualityScorer()
        assert s._completeness_weight == 0.4

    def test_custom_weights(self):
        s = DataQualityScorer(config={"completeness_weight": 0.5})
        assert s._completeness_weight == 0.5

    def test_initial_statistics(self):
        s = DataQualityScorer()
        assert s.get_statistics()["files_scored"] == 0


class TestScoreFile:
    def test_score_complete_data(self):
        s = DataQualityScorer()
        result = s.score_file(["a", "b"], [["1", "2"], ["3", "4"]])
        assert result["overall_score"] > 0.5

    def test_score_with_nulls(self):
        s = DataQualityScorer()
        result = s.score_file(["a", "b"], [["1", None], [None, "4"]])
        assert result["overall_score"] < 1.0
        assert result["completeness_score"] < 1.0

    def test_score_empty_data(self):
        s = DataQualityScorer()
        result = s.score_file(["a"], [])
        # completeness=0.0, accuracy=0.0, consistency=1.0 (< 2 rows)
        # overall = 0.0*0.4 + 0.0*0.35 + 1.0*0.25 = 0.25
        assert result["overall_score"] == pytest.approx(0.25, rel=1e-2)

    def test_score_updates_stats(self):
        s = DataQualityScorer()
        s.score_file(["a"], [["1"], ["2"]])
        assert s.get_statistics()["files_scored"] == 1
        assert s.get_statistics()["rows_scored"] == 2


class TestScoreCompleteness:
    def test_100_percent(self):
        s = DataQualityScorer()
        score = s.score_completeness(["a", "b"], [["1", "2"], ["3", "4"]])
        assert score == 1.0

    def test_50_percent(self):
        s = DataQualityScorer()
        score = s.score_completeness(["a", "b"], [["1", None], [None, "4"]])
        assert score == pytest.approx(0.5, rel=1e-2)

    def test_empty(self):
        s = DataQualityScorer()
        assert s.score_completeness([], []) == 0.0

    def test_all_null(self):
        s = DataQualityScorer()
        score = s.score_completeness(["a", "b"], [[None, None], [None, None]])
        assert score == 0.0

    def test_partial(self):
        s = DataQualityScorer()
        score = s.score_completeness(["a", "b", "c"], [["1", "2", "3"], ["4", None, ""]])
        assert 0.0 < score < 1.0


class TestScoreAccuracy:
    def test_all_valid(self):
        s = DataQualityScorer()
        score = s.score_accuracy(["a"], [["1"], ["2"]])
        assert score == 1.0

    def test_with_nulls(self):
        s = DataQualityScorer()
        score = s.score_accuracy(["a"], [[None], ["2"]])
        assert score < 1.0

    def test_empty(self):
        s = DataQualityScorer()
        assert s.score_accuracy(["a"], []) == 0.0


class TestScoreConsistency:
    def test_consistent_types(self):
        s = DataQualityScorer()
        score = s.score_consistency(["a"], [["x"], ["y"], ["z"]])
        assert score == 1.0

    def test_mixed_types(self):
        s = DataQualityScorer()
        score = s.score_consistency(["a"], [["text"], [42], [3.14]])
        assert score < 1.0

    def test_single_row(self):
        s = DataQualityScorer()
        assert s.score_consistency(["a"], [["x"]]) == 1.0


class TestScoreColumn:
    def test_all_filled(self):
        s = DataQualityScorer()
        assert s.score_column(["a", "b", "c"]) == 1.0

    def test_half_filled(self):
        s = DataQualityScorer()
        assert s.score_column(["a", None]) == 0.5

    def test_all_empty(self):
        s = DataQualityScorer()
        assert s.score_column([None, None]) == 0.0

    def test_empty_strings(self):
        s = DataQualityScorer()
        assert s.score_column(["", ""]) == 0.0


class TestDetectOutliers:
    def test_iqr_no_outliers(self):
        s = DataQualityScorer()
        values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        assert s.detect_outliers(values, "iqr") == []

    def test_iqr_with_outlier(self):
        s = DataQualityScorer()
        values = [10.0, 11.0, 12.0, 13.0, 14.0, 100.0]
        outliers = s.detect_outliers(values, "iqr")
        assert 5 in outliers

    def test_zscore_no_outliers(self):
        s = DataQualityScorer()
        values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        assert s.detect_outliers(values, "zscore") == []

    def test_zscore_with_outlier(self):
        s = DataQualityScorer()
        # For z-score > 3.0, we need many tightly clustered values plus one extreme.
        # 20 values near 10.0 + one at 10000.0 ensures the outlier's z-score >> 3.
        values = [10.0] * 20 + [10000.0]
        outliers = s.detect_outliers(values, "zscore")
        assert 20 in outliers

    def test_too_few_values(self):
        s = DataQualityScorer()
        assert s.detect_outliers([1.0, 2.0]) == []


class TestDetectDuplicates:
    def test_no_duplicates(self):
        s = DataQualityScorer()
        rows = [["a", "1"], ["b", "2"], ["c", "3"]]
        assert s.detect_duplicates(rows) == []

    def test_exact_duplicates(self):
        s = DataQualityScorer()
        rows = [["a", "1"], ["b", "2"], ["a", "1"]]
        dups = s.detect_duplicates(rows, "exact")
        assert 2 in dups

    def test_key_based_duplicates(self):
        s = DataQualityScorer()
        rows = [["a", "1"], ["b", "2"], ["a", "3"]]
        dups = s.detect_duplicates(rows, "key", key_columns=[0])
        assert 2 in dups

    def test_all_duplicates(self):
        s = DataQualityScorer()
        rows = [["a"], ["a"], ["a"]]
        dups = s.detect_duplicates(rows)
        assert len(dups) == 2


class TestComputeQualityLevel:
    def test_excellent(self):
        s = DataQualityScorer()
        assert s.compute_quality_level(0.95) == "excellent"

    def test_good(self):
        s = DataQualityScorer()
        assert s.compute_quality_level(0.75) == "good"

    def test_fair(self):
        s = DataQualityScorer()
        assert s.compute_quality_level(0.55) == "fair"

    def test_poor(self):
        s = DataQualityScorer()
        assert s.compute_quality_level(0.3) == "poor"

    def test_boundary_excellent(self):
        s = DataQualityScorer()
        assert s.compute_quality_level(0.9) == "excellent"

    def test_boundary_good(self):
        s = DataQualityScorer()
        assert s.compute_quality_level(0.7) == "good"

    def test_boundary_fair(self):
        s = DataQualityScorer()
        assert s.compute_quality_level(0.5) == "fair"


class TestGenerateIssues:
    def test_null_issues(self):
        s = DataQualityScorer()
        issues = s.generate_issues(["a", "b"], [["1", None], ["2", None]])
        null_issues = [i for i in issues if i["type"] == "null_values"]
        assert len(null_issues) >= 1

    def test_duplicate_issues(self):
        s = DataQualityScorer()
        issues = s.generate_issues(["a"], [["1"], ["1"]])
        dup_issues = [i for i in issues if i["type"] == "duplicates"]
        assert len(dup_issues) >= 1

    def test_no_issues(self):
        s = DataQualityScorer()
        issues = s.generate_issues(["a"], [["1"], ["2"]])
        null_issues = [i for i in issues if i["type"] == "null_values"]
        assert len(null_issues) == 0


class TestWeightedScoring:
    def test_weights_affect_overall(self):
        # Use data where completeness != accuracy so different weights produce different scores.
        # Row with empty string counts as incomplete but non-None, so accuracy stays 1.0 while completeness < 1.0
        s1 = DataQualityScorer(config={"completeness_weight": 1.0, "accuracy_weight": 0.0, "consistency_weight": 0.0})
        s2 = DataQualityScorer(config={"completeness_weight": 0.0, "accuracy_weight": 0.0, "consistency_weight": 1.0})
        # Data with mixed types in column a (str vs int) -> consistency < 1.0
        # All cells filled and non-None -> completeness = 1.0
        data = [["text", "x"], [42, "y"], [3.14, "z"]]
        r1 = s1.score_file(["a", "b"], data)
        r2 = s2.score_file(["a", "b"], data)
        # s1 overall = completeness (1.0), s2 overall = consistency (< 1.0 due to mixed types)
        assert r1["overall_score"] != r2["overall_score"]


class TestDataQualityScorerStatistics:
    def test_stats_accumulate(self):
        s = DataQualityScorer()
        s.score_file(["a"], [["1"], ["2"]])
        s.score_file(["b", "c"], [["x", "y"]])
        stats = s.get_statistics()
        assert stats["files_scored"] == 2
        assert stats["rows_scored"] == 3
