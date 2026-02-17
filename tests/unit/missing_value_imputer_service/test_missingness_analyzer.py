# -*- coding: utf-8 -*-
"""
Unit tests for MissingnessAnalyzerEngine - AGENT-DATA-012

Tests analyze_dataset, classify_missingness, compute_pattern_matrix,
get_column_analysis, compute_missing_correlations, detect_pattern_type,
recommend_strategies, helper functions, and edge cases.
Target: 60+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
"""

from __future__ import annotations

import math

import pytest

from greenlang.missing_value_imputer.config import MissingValueImputerConfig
from greenlang.missing_value_imputer.missingness_analyzer import (
    MissingnessAnalyzerEngine,
    _is_missing,
    _is_numeric,
    _safe_mean,
    _safe_median,
    _safe_stdev,
    _detect_column_type,
    _pearson_correlation,
)
from greenlang.missing_value_imputer.models import (
    DataColumnType,
    ImputationStrategy,
    MissingnessType,
    PatternType,
)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestIsMissing:
    def test_none_is_missing(self):
        assert _is_missing(None) is True

    def test_empty_string_is_missing(self):
        assert _is_missing("") is True

    def test_whitespace_is_missing(self):
        assert _is_missing("   ") is True

    def test_nan_is_missing(self):
        assert _is_missing(float("nan")) is True

    def test_zero_is_not_missing(self):
        assert _is_missing(0) is False

    def test_false_is_not_missing(self):
        assert _is_missing(False) is False

    def test_string_is_not_missing(self):
        assert _is_missing("hello") is False

    def test_number_is_not_missing(self):
        assert _is_missing(42.0) is False


class TestIsNumeric:
    def test_int_is_numeric(self):
        assert _is_numeric(42) is True

    def test_float_is_numeric(self):
        assert _is_numeric(3.14) is True

    def test_bool_is_not_numeric(self):
        assert _is_numeric(True) is False

    def test_string_is_not_numeric(self):
        assert _is_numeric("42") is False

    def test_none_is_not_numeric(self):
        assert _is_numeric(None) is False


class TestSafeFunctions:
    def test_safe_mean_empty(self):
        assert _safe_mean([]) == 0.0

    def test_safe_mean_values(self):
        assert _safe_mean([2.0, 4.0]) == 3.0

    def test_safe_median_empty(self):
        assert _safe_median([]) == 0.0

    def test_safe_median_values(self):
        assert _safe_median([1.0, 2.0, 3.0]) == 2.0

    def test_safe_stdev_single(self):
        assert _safe_stdev([1.0]) == 0.0

    def test_safe_stdev_values(self):
        result = _safe_stdev([1.0, 2.0, 3.0])
        assert result > 0


class TestDetectColumnType:
    def test_numeric(self):
        assert _detect_column_type([1, 2, 3]) == DataColumnType.NUMERIC

    def test_text(self):
        assert _detect_column_type(["a", "b"]) == DataColumnType.TEXT

    def test_boolean(self):
        assert _detect_column_type([True, False, True]) == DataColumnType.BOOLEAN

    def test_empty(self):
        assert _detect_column_type([]) == DataColumnType.TEXT


class TestPearsonCorrelation:
    def test_perfect_positive(self):
        r = _pearson_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        assert abs(r - 1.0) < 1e-6

    def test_perfect_negative(self):
        r = _pearson_correlation([1.0, 2.0, 3.0], [6.0, 4.0, 2.0])
        assert abs(r - (-1.0)) < 1e-6

    def test_insufficient_data(self):
        assert _pearson_correlation([1.0], [2.0]) == 0.0

    def test_zero_variance(self):
        assert _pearson_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0


# ---------------------------------------------------------------------------
# MissingnessAnalyzerEngine tests
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return MissingnessAnalyzerEngine(MissingValueImputerConfig())


class TestAnalyzeDataset:
    def test_raises_on_empty(self, engine):
        with pytest.raises(ValueError, match="non-empty"):
            engine.analyze_dataset([])

    def test_complete_data(self, engine):
        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        report = engine.analyze_dataset(records)
        assert report.total_records == 2
        assert report.columns_with_missing == 0
        assert report.complete_record_pct == 1.0

    def test_missing_data(self, engine, sample_records_with_missing):
        report = engine.analyze_dataset(sample_records_with_missing)
        assert report.total_records == 10
        assert report.columns_with_missing > 0
        assert report.complete_record_pct < 1.0

    def test_provenance_hash_set(self, engine):
        records = [{"x": 1}, {"x": None}]
        report = engine.analyze_dataset(records)
        assert report.provenance_hash is not None
        assert len(report.provenance_hash) == 64

    def test_specific_columns(self, engine):
        records = [{"a": 1, "b": None, "c": "x"}, {"a": 2, "b": 3, "c": "y"}]
        report = engine.analyze_dataset(records, columns=["b"])
        assert report.total_columns == 1

    def test_pattern_is_set(self, engine, sample_records_with_missing):
        report = engine.analyze_dataset(sample_records_with_missing)
        assert report.pattern is not None
        assert report.pattern.total_missing > 0


class TestGetColumnAnalysis:
    def test_raises_on_empty(self, engine):
        with pytest.raises(ValueError, match="non-empty"):
            engine.get_column_analysis([], "col")

    def test_numeric_column(self, engine):
        records = [{"val": 1.0}, {"val": 2.0}, {"val": None}, {"val": 4.0}]
        ca = engine.get_column_analysis(records, "val")
        assert ca.column_name == "val"
        assert ca.total_values == 4
        assert ca.missing_count == 1
        assert ca.column_type == DataColumnType.NUMERIC
        assert ca.mean_value is not None

    def test_text_column(self, engine):
        records = [{"cat": "a"}, {"cat": "b"}, {"cat": None}]
        ca = engine.get_column_analysis(records, "cat")
        assert ca.column_type == DataColumnType.TEXT
        assert ca.missing_count == 1

    def test_no_missing(self, engine):
        records = [{"x": 1}, {"x": 2}]
        ca = engine.get_column_analysis(records, "x")
        assert ca.missing_count == 0
        assert ca.missing_pct == 0.0

    def test_all_missing(self, engine):
        records = [{"x": None}, {"x": None}]
        ca = engine.get_column_analysis(records, "x")
        assert ca.missing_count == 2
        assert ca.missing_pct == 1.0

    def test_provenance_hash(self, engine):
        records = [{"x": 1}, {"x": None}]
        ca = engine.get_column_analysis(records, "x")
        assert len(ca.provenance_hash) == 64


class TestClassifyMissingness:
    def test_empty_data_returns_unknown(self, engine):
        result = engine.classify_missingness([], [])
        assert result == MissingnessType.UNKNOWN

    def test_no_missing_returns_unknown(self, engine):
        data = [1, 2, 3, 4, 5]
        full = [{"a": v} for v in data]
        result = engine.classify_missingness(data, full)
        assert result == MissingnessType.UNKNOWN

    def test_all_missing_returns_unknown(self, engine):
        data = [None, None, None]
        full = [{"a": None} for _ in data]
        result = engine.classify_missingness(data, full)
        assert result == MissingnessType.UNKNOWN

    def test_small_dataset_defaults_mcar(self, engine):
        data = [1, None, 3, None, 5]
        full = [{"a": v, "b": i} for i, v in enumerate(data)]
        result = engine.classify_missingness(data, full)
        assert result in {MissingnessType.MCAR, MissingnessType.UNKNOWN}


class TestComputePatternMatrix:
    def test_empty_records(self, engine):
        result = engine.compute_pattern_matrix([])
        assert result["n_patterns"] == 0
        assert result["patterns"] == []

    def test_simple_patterns(self, engine):
        records = [
            {"a": 1, "b": 2},
            {"a": None, "b": 3},
            {"a": 1, "b": None},
        ]
        result = engine.compute_pattern_matrix(records)
        assert result["n_patterns"] >= 2
        assert "columns" in result
        assert "provenance_hash" in result

    def test_all_complete(self, engine):
        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = engine.compute_pattern_matrix(records)
        assert result["n_patterns"] == 1
        assert result["patterns"][0]["pattern"] == [1, 1]


class TestDetectPatternType:
    def test_empty_records(self, engine):
        result = engine.detect_pattern_type([])
        assert result == PatternType.ARBITRARY

    def test_no_missing(self, engine):
        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = engine.detect_pattern_type(records)
        assert result == PatternType.ARBITRARY

    def test_univariate(self, engine):
        records = [
            {"a": 1, "b": 2},
            {"a": None, "b": 3},
            {"a": 1, "b": 4},
        ]
        result = engine.detect_pattern_type(records)
        assert result == PatternType.UNIVARIATE

    def test_planned(self, engine):
        records = [
            {"a": 1, "b": 2, "c": 3},
            {"a": None, "b": None, "c": None},
            {"a": 4, "b": 5, "c": 6},
        ]
        result = engine.detect_pattern_type(records)
        assert result == PatternType.PLANNED


class TestComputeMissingCorrelations:
    def test_empty_records(self, engine):
        assert engine.compute_missing_correlations([]) == {}

    def test_single_column(self, engine):
        records = [{"a": 1}, {"a": None}]
        assert engine.compute_missing_correlations(records) == {}

    def test_two_columns(self, engine):
        records = [
            {"a": 1, "b": 2},
            {"a": None, "b": None},
            {"a": 3, "b": 4},
        ]
        corr = engine.compute_missing_correlations(records)
        assert "a" in corr
        assert "b" in corr
        assert corr["a"]["a"] == 1.0
        assert corr["b"]["b"] == 1.0


class TestRecommendStrategies:
    def test_no_missing_columns(self, engine):
        records = [{"a": 1}, {"a": 2}]
        report = engine.analyze_dataset(records)
        strategies = engine.recommend_strategies(report)
        assert len(strategies) == 0

    def test_returns_strategy_for_missing(self, engine, sample_records_with_missing):
        report = engine.analyze_dataset(sample_records_with_missing)
        strategies = engine.recommend_strategies(report)
        assert len(strategies) > 0
        for col, sel in strategies.items():
            assert sel.column_name == col
            assert sel.recommended_strategy is not None
            assert sel.estimated_confidence > 0
            assert len(sel.provenance_hash) == 64

    def test_numeric_mcar_recommends_median(self, engine):
        records = [{"x": float(i)} for i in range(20)]
        records[3]["x"] = None
        report = engine.analyze_dataset(records)
        strategies = engine.recommend_strategies(report)
        if "x" in strategies:
            assert strategies["x"].recommended_strategy in {
                ImputationStrategy.MEDIAN,
                ImputationStrategy.MEAN,
                ImputationStrategy.KNN,
            }


class TestRecommendColumnStrategy:
    """Test the private _recommend_column_strategy decision tree."""

    def test_high_missing_ml_enabled(self, engine):
        strategy = engine._recommend_column_strategy(
            DataColumnType.NUMERIC, MissingnessType.MCAR, 0.6, 200
        )
        assert strategy == ImputationStrategy.MICE

    def test_high_missing_ml_disabled(self):
        cfg = MissingValueImputerConfig(enable_ml_imputation=False)
        eng = MissingnessAnalyzerEngine(cfg)
        strategy = eng._recommend_column_strategy(
            DataColumnType.NUMERIC, MissingnessType.MCAR, 0.6, 200
        )
        assert strategy == ImputationStrategy.REGULATORY_DEFAULT

    def test_mnar_numeric(self, engine):
        strategy = engine._recommend_column_strategy(
            DataColumnType.NUMERIC, MissingnessType.MNAR, 0.1, 100
        )
        assert strategy == ImputationStrategy.REGRESSION

    def test_mnar_categorical(self, engine):
        strategy = engine._recommend_column_strategy(
            DataColumnType.CATEGORICAL, MissingnessType.MNAR, 0.1, 100
        )
        assert strategy == ImputationStrategy.RULE_BASED

    def test_mar_numeric(self, engine):
        strategy = engine._recommend_column_strategy(
            DataColumnType.NUMERIC, MissingnessType.MAR, 0.1, 100
        )
        assert strategy == ImputationStrategy.KNN

    def test_mcar_numeric(self, engine):
        strategy = engine._recommend_column_strategy(
            DataColumnType.NUMERIC, MissingnessType.MCAR, 0.1, 100
        )
        assert strategy == ImputationStrategy.MEDIAN

    def test_mcar_categorical(self, engine):
        strategy = engine._recommend_column_strategy(
            DataColumnType.CATEGORICAL, MissingnessType.MCAR, 0.1, 100
        )
        assert strategy == ImputationStrategy.MODE

    def test_datetime_with_timeseries(self, engine):
        strategy = engine._recommend_column_strategy(
            DataColumnType.DATETIME, MissingnessType.MCAR, 0.1, 100
        )
        assert strategy == ImputationStrategy.LINEAR_INTERPOLATION

    def test_datetime_without_timeseries(self):
        cfg = MissingValueImputerConfig(enable_timeseries=False)
        eng = MissingnessAnalyzerEngine(cfg)
        strategy = eng._recommend_column_strategy(
            DataColumnType.DATETIME, MissingnessType.MCAR, 0.1, 100
        )
        assert strategy == ImputationStrategy.LOCF

    def test_text_column(self, engine):
        strategy = engine._recommend_column_strategy(
            DataColumnType.TEXT, MissingnessType.MCAR, 0.1, 100
        )
        assert strategy == ImputationStrategy.MODE


class TestEstimateConfidence:
    def test_low_missing_mcar(self, engine):
        c = engine._estimate_confidence(ImputationStrategy.MEDIAN, 0.05, MissingnessType.MCAR)
        assert c > 0.7

    def test_high_missing_penalty(self, engine):
        c_low = engine._estimate_confidence(ImputationStrategy.MEAN, 0.1, MissingnessType.MCAR)
        c_high = engine._estimate_confidence(ImputationStrategy.MEAN, 0.6, MissingnessType.MCAR)
        assert c_high < c_low

    def test_mnar_penalty(self, engine):
        c_mcar = engine._estimate_confidence(ImputationStrategy.MEAN, 0.1, MissingnessType.MCAR)
        c_mnar = engine._estimate_confidence(ImputationStrategy.MEAN, 0.1, MissingnessType.MNAR)
        assert c_mnar < c_mcar

    def test_max_one(self, engine):
        c = engine._estimate_confidence(ImputationStrategy.LOOKUP_TABLE, 0.01, MissingnessType.MCAR)
        assert c <= 1.0


class TestAggregeMissingnessType:
    def test_empty_returns_unknown(self, engine):
        assert engine._aggregate_missingness_type([]) == MissingnessType.UNKNOWN

    def test_mnar_takes_priority(self, engine):
        types = [MissingnessType.MCAR, MissingnessType.MNAR, MissingnessType.MAR]
        assert engine._aggregate_missingness_type(types) == MissingnessType.MNAR

    def test_mar_over_mcar(self, engine):
        types = [MissingnessType.MCAR, MissingnessType.MAR]
        assert engine._aggregate_missingness_type(types) == MissingnessType.MAR

    def test_all_unknown(self, engine):
        types = [MissingnessType.UNKNOWN, MissingnessType.UNKNOWN]
        assert engine._aggregate_missingness_type(types) == MissingnessType.UNKNOWN
