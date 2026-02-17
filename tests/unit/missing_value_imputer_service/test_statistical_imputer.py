# -*- coding: utf-8 -*-
"""
Unit tests for StatisticalImputerEngine - AGENT-DATA-012

Tests impute_mean, impute_median, impute_mode, impute_knn,
impute_regression, impute_hot_deck, impute_locf, impute_nocb,
impute_grouped, confidence scoring, and edge cases.
Target: 60+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
"""

from __future__ import annotations

import pytest

from greenlang.missing_value_imputer.config import MissingValueImputerConfig
from greenlang.missing_value_imputer.models import ImputationStrategy
from greenlang.missing_value_imputer.statistical_imputer import StatisticalImputerEngine


@pytest.fixture
def engine():
    return StatisticalImputerEngine(MissingValueImputerConfig())


@pytest.fixture
def numeric_records():
    return [
        {"val": 10.0, "feat": 1.0},
        {"val": 20.0, "feat": 2.0},
        {"val": None, "feat": 3.0},
        {"val": 40.0, "feat": 4.0},
        {"val": 50.0, "feat": 5.0},
    ]


@pytest.fixture
def categorical_records():
    return [
        {"cat": "A", "grp": "x"},
        {"cat": "B", "grp": "x"},
        {"cat": "A", "grp": "y"},
        {"cat": None, "grp": "x"},
        {"cat": "A", "grp": "y"},
    ]


# ---------------------------------------------------------------------------
# Mean imputation
# ---------------------------------------------------------------------------


class TestImputeMean:
    def test_basic_mean(self, engine, numeric_records):
        result = engine.impute_mean(numeric_records, "val")
        assert len(result) == 1
        expected_mean = (10 + 20 + 40 + 50) / 4
        assert result[0].imputed_value == pytest.approx(expected_mean, rel=1e-4)
        assert result[0].strategy == ImputationStrategy.MEAN

    def test_raises_on_no_numeric(self, engine):
        records = [{"x": "a"}, {"x": None}]
        with pytest.raises(ValueError, match="No numeric"):
            engine.impute_mean(records, "x")

    def test_all_missing_raises(self, engine):
        records = [{"x": None}, {"x": None}]
        with pytest.raises(ValueError, match="No numeric"):
            engine.impute_mean(records, "x")

    def test_no_missing_returns_empty(self, engine):
        records = [{"x": 1.0}, {"x": 2.0}]
        result = engine.impute_mean(records, "x")
        assert result == []

    def test_provenance_hash(self, engine, numeric_records):
        result = engine.impute_mean(numeric_records, "val")
        assert len(result[0].provenance_hash) == 64

    def test_confidence_is_set(self, engine, numeric_records):
        result = engine.impute_mean(numeric_records, "val")
        assert 0.0 < result[0].confidence <= 1.0

    def test_record_index(self, engine, numeric_records):
        result = engine.impute_mean(numeric_records, "val")
        assert result[0].record_index == 2


# ---------------------------------------------------------------------------
# Median imputation
# ---------------------------------------------------------------------------


class TestImputeMedian:
    def test_basic_median(self, engine, numeric_records):
        result = engine.impute_median(numeric_records, "val")
        assert len(result) == 1
        # median of [10, 20, 40, 50] = 30
        assert result[0].imputed_value == pytest.approx(30.0, rel=1e-4)
        assert result[0].strategy == ImputationStrategy.MEDIAN

    def test_raises_on_no_numeric(self, engine):
        records = [{"x": "a"}, {"x": None}]
        with pytest.raises(ValueError, match="No numeric"):
            engine.impute_median(records, "x")

    def test_single_value(self, engine):
        records = [{"x": 5.0}, {"x": None}]
        result = engine.impute_median(records, "x")
        assert result[0].imputed_value == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Mode imputation
# ---------------------------------------------------------------------------


class TestImputeMode:
    def test_basic_mode(self, engine, categorical_records):
        result = engine.impute_mode(categorical_records, "cat")
        assert len(result) == 1
        assert result[0].imputed_value == "A"
        assert result[0].strategy == ImputationStrategy.MODE

    def test_raises_on_all_missing(self, engine):
        records = [{"x": None}, {"x": None}]
        with pytest.raises(ValueError, match="No non-missing"):
            engine.impute_mode(records, "x")

    def test_numeric_mode(self, engine):
        records = [{"x": 1}, {"x": 1}, {"x": 2}, {"x": None}]
        result = engine.impute_mode(records, "x")
        assert result[0].imputed_value == 1

    def test_confidence_based_on_dominance(self, engine):
        records = [{"x": "A"}] * 9 + [{"x": "B"}, {"x": None}]
        result = engine.impute_mode(records, "x")
        assert result[0].confidence > 0.8


# ---------------------------------------------------------------------------
# KNN imputation
# ---------------------------------------------------------------------------


class TestImputeKnn:
    def test_basic_knn(self, engine):
        records = [
            {"val": 10.0, "f1": 1.0, "f2": 1.0},
            {"val": 20.0, "f1": 2.0, "f2": 2.0},
            {"val": 30.0, "f1": 3.0, "f2": 3.0},
            {"val": 40.0, "f1": 4.0, "f2": 4.0},
            {"val": 50.0, "f1": 5.0, "f2": 5.0},
            {"val": None, "f1": 3.0, "f2": 3.0},
        ]
        result = engine.impute_knn(records, "val", k=3)
        assert len(result) == 1
        assert result[0].strategy == ImputationStrategy.KNN
        # nearest to (3,3) are vals 20, 30, 40
        assert 20.0 <= result[0].imputed_value <= 40.0

    def test_raises_insufficient_complete(self, engine):
        records = [
            {"val": 1.0, "f": 1.0},
            {"val": None, "f": 2.0},
        ]
        with pytest.raises(ValueError, match="at least"):
            engine.impute_knn(records, "val", k=5)

    def test_raises_no_feature_columns(self, engine):
        records = [{"val": 1.0}, {"val": None}]
        with pytest.raises(ValueError, match="No numeric feature"):
            engine.impute_knn(records, "val")

    def test_defaults_to_config_k(self):
        cfg = MissingValueImputerConfig(knn_neighbors=3)
        eng = StatisticalImputerEngine(cfg)
        records = [
            {"val": float(i), "f": float(i)} for i in range(10)
        ]
        records[5]["val"] = None
        result = eng.impute_knn(records, "val")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Regression imputation
# ---------------------------------------------------------------------------


class TestImputeRegression:
    def test_basic_regression(self, engine):
        records = [
            {"y": 2.0, "x": 1.0},
            {"y": 4.0, "x": 2.0},
            {"y": 6.0, "x": 3.0},
            {"y": 8.0, "x": 4.0},
            {"y": None, "x": 5.0},
        ]
        result = engine.impute_regression(records, "y")
        assert len(result) == 1
        assert result[0].strategy == ImputationStrategy.REGRESSION
        # y = 2x -> expect ~10
        assert result[0].imputed_value == pytest.approx(10.0, rel=0.1)

    def test_raises_no_predictors(self, engine):
        records = [{"y": 1.0}, {"y": None}]
        with pytest.raises(ValueError, match="No predictor"):
            engine.impute_regression(records, "y")

    def test_raises_insufficient_records(self, engine):
        records = [
            {"y": 1.0, "x": 1.0},
            {"y": None, "x": 2.0},
        ]
        with pytest.raises(ValueError, match="Need at least"):
            engine.impute_regression(records, "y")

    def test_custom_predictors(self, engine):
        records = [
            {"y": 1.0, "a": 1.0, "b": 0.0},
            {"y": 2.0, "a": 2.0, "b": 0.0},
            {"y": 3.0, "a": 3.0, "b": 0.0},
            {"y": 4.0, "a": 4.0, "b": 0.0},
            {"y": None, "a": 5.0, "b": 0.0},
        ]
        result = engine.impute_regression(records, "y", predictor_columns=["a"])
        assert len(result) == 1
        assert result[0].imputed_value == pytest.approx(5.0, rel=0.1)


# ---------------------------------------------------------------------------
# Hot-deck imputation
# ---------------------------------------------------------------------------


class TestImputeHotDeck:
    def test_random_hot_deck(self, engine):
        records = [{"x": 10}, {"x": 20}, {"x": None}, {"x": 30}]
        result = engine.impute_hot_deck(records, "x", method="random")
        assert len(result) == 1
        assert result[0].imputed_value in {10, 20, 30}
        assert result[0].strategy == ImputationStrategy.HOT_DECK

    def test_sequential_hot_deck(self, engine):
        records = [{"x": 10}, {"x": 20}, {"x": None}, {"x": 30}]
        result = engine.impute_hot_deck(records, "x", method="sequential")
        assert len(result) == 1
        # nearest donor to index 2 is index 1 (val 20) or 3 (val 30)
        assert result[0].imputed_value in {20, 30}

    def test_raises_no_observed(self, engine):
        records = [{"x": None}, {"x": None}]
        with pytest.raises(ValueError, match="No observed"):
            engine.impute_hot_deck(records, "x")


# ---------------------------------------------------------------------------
# LOCF imputation
# ---------------------------------------------------------------------------


class TestImputeLocf:
    def test_basic_locf(self, engine):
        records = [{"x": 1}, {"x": None}, {"x": None}, {"x": 4}]
        result = engine.impute_locf(records, "x")
        assert len(result) == 2
        assert result[0].imputed_value == 1
        assert result[1].imputed_value == 1
        assert result[0].strategy == ImputationStrategy.LOCF

    def test_leading_missing_not_imputed(self, engine):
        records = [{"x": None}, {"x": 2}, {"x": None}]
        result = engine.impute_locf(records, "x")
        assert len(result) == 1
        assert result[0].record_index == 2

    def test_confidence_decays(self, engine):
        records = [{"x": 1}] + [{"x": None}] * 5
        result = engine.impute_locf(records, "x")
        assert result[0].confidence > result[-1].confidence

    def test_with_sort_column(self, engine):
        records = [
            {"x": None, "t": "b"},
            {"x": 10, "t": "a"},
            {"x": None, "t": "c"},
        ]
        result = engine.impute_locf(records, "x", sort_column="t")
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# NOCB imputation
# ---------------------------------------------------------------------------


class TestImputeNocb:
    def test_basic_nocb(self, engine):
        records = [{"x": None}, {"x": None}, {"x": 3}, {"x": 4}]
        result = engine.impute_nocb(records, "x")
        assert len(result) == 2
        assert result[0].imputed_value == 3
        assert result[1].imputed_value == 3
        assert result[0].strategy == ImputationStrategy.NOCB

    def test_trailing_missing_not_imputed(self, engine):
        records = [{"x": 1}, {"x": None}]
        result = engine.impute_nocb(records, "x")
        assert result == []

    def test_sorted_by_record_index(self, engine):
        records = [{"x": None}, {"x": None}, {"x": 10}]
        result = engine.impute_nocb(records, "x")
        indices = [iv.record_index for iv in result]
        assert indices == sorted(indices)


# ---------------------------------------------------------------------------
# Grouped imputation
# ---------------------------------------------------------------------------


class TestImputeGrouped:
    def test_grouped_mean(self, engine):
        records = [
            {"val": 10.0, "grp": "A"},
            {"val": 20.0, "grp": "A"},
            {"val": None, "grp": "A"},
            {"val": 100.0, "grp": "B"},
            {"val": None, "grp": "B"},
        ]
        result = engine.impute_grouped(records, "val", "grp", method="mean")
        assert len(result) == 2
        a_result = [r for r in result if r.record_index == 2][0]
        b_result = [r for r in result if r.record_index == 4][0]
        assert a_result.imputed_value == pytest.approx(15.0, rel=1e-4)
        assert b_result.imputed_value == pytest.approx(100.0, rel=1e-4)

    def test_grouped_median(self, engine):
        records = [
            {"val": 10.0, "grp": "A"},
            {"val": 20.0, "grp": "A"},
            {"val": 30.0, "grp": "A"},
            {"val": None, "grp": "A"},
        ]
        result = engine.impute_grouped(records, "val", "grp", method="median")
        assert result[0].imputed_value == pytest.approx(20.0, rel=1e-4)

    def test_grouped_mode(self, engine):
        records = [
            {"val": "X", "grp": "A"},
            {"val": "X", "grp": "A"},
            {"val": "Y", "grp": "A"},
            {"val": None, "grp": "A"},
        ]
        result = engine.impute_grouped(records, "val", "grp", method="mode")
        assert result[0].imputed_value == "X"

    def test_fallback_to_global(self, engine):
        records = [
            {"val": 10.0, "grp": "A"},
            {"val": 20.0, "grp": "A"},
            {"val": None, "grp": "B"},
        ]
        result = engine.impute_grouped(records, "val", "grp", method="mean")
        assert len(result) == 1
        assert result[0].imputed_value == pytest.approx(15.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Confidence computation
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_base_confidences(self, engine):
        c_mean = engine._compute_confidence("mean", 100, 0.0)
        c_median = engine._compute_confidence("median", 100, 0.0)
        c_knn = engine._compute_confidence("knn", 100, 0.0)
        assert c_median > c_mean
        assert c_knn > c_median

    def test_sample_size_adjustment(self, engine):
        c_large = engine._adjust_confidence_by_sample(0.80, 1000)
        c_small = engine._adjust_confidence_by_sample(0.80, 5)
        assert c_large > c_small

    def test_very_small_sample(self, engine):
        c = engine._adjust_confidence_by_sample(0.80, 2)
        assert c < 0.80

    def test_confidence_clamp(self, engine):
        c = engine._compute_confidence("mean", 10000, 0.0)
        assert c <= 1.0
        assert c >= 0.0

    def test_variance_penalty(self, engine):
        c_low_var = engine._compute_confidence("mean", 100, 1.0)
        c_high_var = engine._compute_confidence("mean", 100, 200.0)
        assert c_low_var > c_high_var
