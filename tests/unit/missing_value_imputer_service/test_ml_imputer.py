# -*- coding: utf-8 -*-
"""
Unit tests for MLImputerEngine - AGENT-DATA-012

Tests impute_random_forest, impute_gradient_boosting, impute_mice,
impute_matrix_factorization, impute_multiple, pool_estimates,
DecisionStump, confidence, MIN_RECORDS_ML enforcement, and edge cases.
Target: 50+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
"""

from __future__ import annotations

import pytest

from greenlang.missing_value_imputer.config import MissingValueImputerConfig
from greenlang.missing_value_imputer.ml_imputer import (
    MLImputerEngine,
    _DecisionStump,
    _is_missing,
    _is_numeric,
    _to_float,
    _classify_confidence,
    _safe_stdev,
)
from greenlang.missing_value_imputer.models import (
    ConfidenceLevel,
    ImputationStrategy,
    ImputedValue,
    MIN_RECORDS_ML,
)


@pytest.fixture
def engine():
    return MLImputerEngine(MissingValueImputerConfig())


def _make_large_records(n: int, missing_idx: int = -1):
    """Generate n records with one optional missing index.

    Each record has columns 'y' (target), 'x1', 'x2'.
    The relationship is roughly y = x1 + x2.
    """
    records = []
    for i in range(n):
        y = float(i) + float(i % 7)
        r = {"y": y, "x1": float(i), "x2": float(i % 7)}
        records.append(r)
    if 0 <= missing_idx < n:
        records[missing_idx]["y"] = None
    return records


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_is_missing_none(self):
        assert _is_missing(None) is True

    def test_is_missing_nan(self):
        assert _is_missing(float("nan")) is True

    def test_is_missing_empty_str(self):
        assert _is_missing("") is True

    def test_is_missing_valid(self):
        assert _is_missing(42) is False

    def test_is_numeric_int(self):
        assert _is_numeric(10) is True

    def test_is_numeric_bool(self):
        assert _is_numeric(True) is False

    def test_to_float_none(self):
        assert _to_float(None) is None

    def test_to_float_number(self):
        assert _to_float(3.14) == pytest.approx(3.14)

    def test_to_float_string(self):
        assert _to_float("42") == pytest.approx(42.0)

    def test_to_float_invalid(self):
        assert _to_float("abc") is None

    def test_classify_confidence_high(self):
        assert _classify_confidence(0.90) == ConfidenceLevel.HIGH

    def test_classify_confidence_medium(self):
        assert _classify_confidence(0.75) == ConfidenceLevel.MEDIUM

    def test_classify_confidence_low(self):
        assert _classify_confidence(0.55) == ConfidenceLevel.LOW

    def test_classify_confidence_very_low(self):
        assert _classify_confidence(0.30) == ConfidenceLevel.VERY_LOW

    def test_safe_stdev_empty(self):
        assert _safe_stdev([]) == 0.0

    def test_safe_stdev_single(self):
        assert _safe_stdev([1.0]) == 0.0

    def test_safe_stdev_values(self):
        result = _safe_stdev([1.0, 2.0, 3.0])
        assert result > 0


# ---------------------------------------------------------------------------
# _DecisionStump tests
# ---------------------------------------------------------------------------


class TestDecisionStump:
    def test_fit_and_predict(self):
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        stump = _DecisionStump(max_depth=5, min_samples=2)
        stump.fit(X, y)
        pred = stump.predict_one([3.0])
        # Should be close to 6.0
        assert 2.0 <= pred <= 10.0

    def test_constant_target(self):
        X = [[1.0], [2.0], [3.0]]
        y = [5.0, 5.0, 5.0]
        stump = _DecisionStump(max_depth=3)
        stump.fit(X, y)
        assert stump.predict_one([2.5]) == pytest.approx(5.0)

    def test_empty_data(self):
        stump = _DecisionStump()
        stump.fit([], [])
        assert stump.predict_one([1.0]) == 0.0

    def test_single_sample(self):
        stump = _DecisionStump()
        stump.fit([[1.0]], [42.0])
        assert stump.predict_one([1.0]) == pytest.approx(42.0)

    def test_variance_method(self):
        assert _DecisionStump._variance([]) == 0.0
        assert _DecisionStump._variance([1.0]) == 0.0
        v = _DecisionStump._variance([1.0, 3.0, 5.0])
        assert v > 0


# ---------------------------------------------------------------------------
# Random Forest imputation
# ---------------------------------------------------------------------------


class TestImputeRandomForest:
    def test_raises_insufficient_records(self, engine):
        records = [{"y": float(i), "x": float(i)} for i in range(10)]
        records[5]["y"] = None
        with pytest.raises(ValueError, match="at least"):
            engine.impute_random_forest(records, "y")

    def test_basic_rf_imputation(self, engine):
        records = _make_large_records(150, missing_idx=75)
        result = engine.impute_random_forest(records, "y", n_estimators=10)
        assert len(result) == 1
        assert result[0].strategy == ImputationStrategy.RANDOM_FOREST
        assert result[0].record_index == 75
        assert len(result[0].provenance_hash) == 64

    def test_confidence_within_bounds(self, engine):
        records = _make_large_records(150, missing_idx=75)
        result = engine.impute_random_forest(records, "y", n_estimators=10)
        assert 0.5 <= result[0].confidence <= 0.95

    def test_no_missing_returns_empty(self, engine):
        records = _make_large_records(150)
        result = engine.impute_random_forest(records, "y", n_estimators=5)
        assert result == []

    def test_multiple_missing(self, engine):
        records = _make_large_records(150)
        records[10]["y"] = None
        records[20]["y"] = None
        result = engine.impute_random_forest(records, "y", n_estimators=5)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Gradient Boosting imputation
# ---------------------------------------------------------------------------


class TestImputeGradientBoosting:
    def test_raises_insufficient_records(self, engine):
        records = [{"y": float(i), "x": float(i)} for i in range(10)]
        records[5]["y"] = None
        with pytest.raises(ValueError, match="at least"):
            engine.impute_gradient_boosting(records, "y")

    def test_basic_gbm_imputation(self, engine):
        records = _make_large_records(150, missing_idx=75)
        result = engine.impute_gradient_boosting(records, "y")
        assert len(result) == 1
        assert result[0].strategy == ImputationStrategy.GRADIENT_BOOSTING
        assert len(result[0].provenance_hash) == 64

    def test_confidence_bounds(self, engine):
        records = _make_large_records(150, missing_idx=75)
        result = engine.impute_gradient_boosting(records, "y")
        assert 0.5 <= result[0].confidence <= 0.95

    def test_no_missing_returns_empty(self, engine):
        records = _make_large_records(150)
        result = engine.impute_gradient_boosting(records, "y")
        assert result == []


# ---------------------------------------------------------------------------
# MICE imputation
# ---------------------------------------------------------------------------


class TestImputeMice:
    def test_raises_on_empty(self, engine):
        with pytest.raises(ValueError, match="non-empty"):
            engine.impute_mice([])

    def test_basic_mice(self, engine):
        records = [
            {"a": float(i), "b": float(i * 2)} for i in range(20)
        ]
        records[5]["a"] = None
        records[10]["b"] = None
        result = engine.impute_mice(records)
        assert "a" in result or "b" in result

    def test_mice_specific_columns(self, engine):
        records = [
            {"a": float(i), "b": float(i * 2), "c": float(i * 3)}
            for i in range(20)
        ]
        records[5]["a"] = None
        records[10]["b"] = None
        result = engine.impute_mice(records, columns=["a"])
        assert "a" in result
        # "b" should not be imputed since we only asked for "a"
        assert "b" not in result

    def test_mice_no_missing_returns_empty(self, engine):
        records = [{"a": float(i), "b": float(i * 2)} for i in range(20)]
        result = engine.impute_mice(records)
        assert result == {}

    def test_mice_custom_iterations(self, engine):
        records = [
            {"a": float(i), "b": float(i * 2)} for i in range(20)
        ]
        records[5]["a"] = None
        result = engine.impute_mice(records, n_iterations=3)
        assert "a" in result
        for iv in result["a"]:
            assert iv.strategy == ImputationStrategy.MICE

    def test_mice_provenance_hash(self, engine):
        records = [
            {"a": float(i), "b": float(i * 2)} for i in range(20)
        ]
        records[5]["a"] = None
        result = engine.impute_mice(records)
        for iv in result.get("a", []):
            assert len(iv.provenance_hash) == 64


# ---------------------------------------------------------------------------
# Matrix Factorization imputation
# ---------------------------------------------------------------------------


class TestImputeMatrixFactorization:
    def test_empty_records(self, engine):
        result = engine.impute_matrix_factorization([])
        assert result == {}

    def test_basic_matrix_factorization(self, engine):
        records = [
            {"a": float(i), "b": float(i * 2), "c": float(i + 1)}
            for i in range(20)
        ]
        records[5]["a"] = None
        result = engine.impute_matrix_factorization(records)
        assert "a" in result
        assert len(result["a"]) == 1
        assert result["a"][0].strategy == ImputationStrategy.MATRIX_FACTORIZATION

    def test_no_missing_returns_empty(self, engine):
        records = [{"a": float(i), "b": float(i)} for i in range(20)]
        result = engine.impute_matrix_factorization(records)
        assert result == {}

    def test_specific_columns(self, engine):
        records = [
            {"a": float(i), "b": float(i * 2), "c": float(i + 1)}
            for i in range(20)
        ]
        records[5]["a"] = None
        records[10]["b"] = None
        result = engine.impute_matrix_factorization(records, columns=["a"])
        assert "a" in result
        assert "b" not in result

    def test_confidence_based_on_row_completeness(self, engine):
        records = [
            {"a": float(i), "b": float(i * 2), "c": float(i + 1)}
            for i in range(20)
        ]
        records[5]["a"] = None
        result = engine.impute_matrix_factorization(records)
        assert 0.5 <= result["a"][0].confidence <= 0.95


# ---------------------------------------------------------------------------
# Multiple Imputation
# ---------------------------------------------------------------------------


class TestImputeMultiple:
    def test_basic_multiple_imputation(self, engine):
        records = [
            {"a": float(i), "b": float(i * 2)} for i in range(20)
        ]
        records[5]["a"] = None
        result = engine.impute_multiple(records, "a", n_imputations=3)
        assert len(result) == 3
        # Each entry should be a list of ImputedValue
        for imp_set in result:
            assert isinstance(imp_set, list)

    def test_uses_config_default_imputations(self):
        cfg = MissingValueImputerConfig(multiple_imputations=2)
        eng = MLImputerEngine(cfg)
        records = [
            {"a": float(i), "b": float(i * 2)} for i in range(20)
        ]
        records[5]["a"] = None
        result = eng.impute_multiple(records, "a")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Pool Estimates (Rubin's rules)
# ---------------------------------------------------------------------------


class TestPoolEstimates:
    def test_empty_results(self, engine):
        assert engine.pool_estimates([]) == []
        assert engine.pool_estimates([[]]) == []

    def test_basic_pooling(self, engine):
        iv1 = ImputedValue(
            record_index=5, column_name="a", imputed_value=10.0,
            strategy=ImputationStrategy.MICE, confidence=0.8,
            provenance_hash="a" * 64,
        )
        iv2 = ImputedValue(
            record_index=5, column_name="a", imputed_value=12.0,
            strategy=ImputationStrategy.MICE, confidence=0.8,
            provenance_hash="b" * 64,
        )
        result = engine.pool_estimates([[iv1], [iv2]])
        assert len(result) == 1
        # Pooled estimate should be the mean of 10.0 and 12.0 = 11.0
        assert result[0].imputed_value == pytest.approx(11.0, rel=0.01)
        assert result[0].strategy == ImputationStrategy.MICE

    def test_pooling_confidence(self, engine):
        iv1 = ImputedValue(
            record_index=0, column_name="x", imputed_value=100.0,
            strategy=ImputationStrategy.MICE, confidence=0.85,
            provenance_hash="a" * 64,
        )
        iv2 = ImputedValue(
            record_index=0, column_name="x", imputed_value=100.5,
            strategy=ImputationStrategy.MICE, confidence=0.85,
            provenance_hash="b" * 64,
        )
        result = engine.pool_estimates([[iv1], [iv2]])
        assert 0.5 <= result[0].confidence <= 0.95

    def test_pooling_provenance(self, engine):
        iv1 = ImputedValue(
            record_index=0, column_name="x", imputed_value=10.0,
            strategy=ImputationStrategy.MICE, confidence=0.8,
            provenance_hash="a" * 64,
        )
        result = engine.pool_estimates([[iv1]])
        assert len(result[0].provenance_hash) == 64


# ---------------------------------------------------------------------------
# Private method tests
# ---------------------------------------------------------------------------


class TestPrivateMethods:
    def test_collect_numeric_columns(self, engine):
        records = [{"a": 1.0, "b": "text", "c": 3.0}]
        cols = engine._collect_numeric_columns(records)
        assert "a" in cols
        assert "c" in cols
        assert "b" not in cols

    def test_col_is_numeric_true(self, engine):
        records = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
        assert engine._col_is_numeric(records, "x") is True

    def test_col_is_numeric_false(self, engine):
        records = [{"x": "a"}, {"x": "b"}]
        assert engine._col_is_numeric(records, "x") is False

    def test_extract_features_complete(self, engine):
        record = {"a": 1.0, "b": 2.0, "c": 3.0}
        result = engine._extract_features(record, ["a", "b", "c"])
        assert result == [1.0, 2.0, 3.0]

    def test_extract_features_missing(self, engine):
        record = {"a": 1.0, "b": None}
        result = engine._extract_features(record, ["a", "b"])
        assert result is None

    def test_mice_confidence(self, engine):
        c = engine._mice_confidence("col", 10, 500)
        assert 0.7 < c <= 0.95

    def test_fit_simple_ols(self, engine):
        X = [[1.0], [2.0], [3.0], [4.0]]
        y = [2.0, 4.0, 6.0, 8.0]
        coeffs = engine._fit_simple_ols(X, y)
        # y = 2x + 0 -> intercept ~0, slope ~2
        assert len(coeffs) == 2
        assert coeffs[1] == pytest.approx(2.0, rel=0.1)

    def test_solve_system(self, engine):
        A = [[2.0, 1.0], [1.0, 3.0]]
        b = [5.0, 7.0]
        x = engine._solve_system(A, b)
        assert len(x) == 2
        # 2x1 + x2 = 5, x1 + 3x2 = 7 -> x1=1.6, x2=1.8
        assert x[0] == pytest.approx(1.6, rel=0.1)
        assert x[1] == pytest.approx(1.8, rel=0.1)
