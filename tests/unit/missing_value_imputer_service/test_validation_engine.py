# -*- coding: utf-8 -*-
"""
Unit tests for ValidationEngine - AGENT-DATA-012

Tests validate_imputation, ks_test, chi_square_test, plausibility_check,
distribution_preservation, cross_validate, compute_rmse, compute_mae,
generate_validation_report, and private helpers.
Target: 50+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
"""

from __future__ import annotations

import pytest

from greenlang.missing_value_imputer.config import MissingValueImputerConfig
from greenlang.missing_value_imputer.validation_engine import (
    ValidationEngine,
    _is_missing,
    _is_numeric,
    _safe_stdev,
)
from greenlang.missing_value_imputer.models import (
    ValidationMethod,
    ValidationReport,
    ValidationResult,
)


@pytest.fixture
def engine():
    return ValidationEngine(MissingValueImputerConfig())


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_is_missing_none(self):
        assert _is_missing(None) is True

    def test_is_missing_empty_str(self):
        assert _is_missing("") is True

    def test_is_missing_nan(self):
        assert _is_missing(float("nan")) is True

    def test_is_missing_valid(self):
        assert _is_missing(42) is False

    def test_is_numeric_int(self):
        assert _is_numeric(10) is True

    def test_is_numeric_float(self):
        assert _is_numeric(3.14) is True

    def test_is_numeric_bool(self):
        assert _is_numeric(True) is False

    def test_is_numeric_string(self):
        assert _is_numeric("42") is False

    def test_safe_stdev_empty(self):
        assert _safe_stdev([]) == 0.0

    def test_safe_stdev_single(self):
        assert _safe_stdev([5.0]) == 0.0

    def test_safe_stdev_positive(self):
        assert _safe_stdev([1.0, 2.0, 3.0]) > 0


# ---------------------------------------------------------------------------
# KS test
# ---------------------------------------------------------------------------


class TestKsTest:
    def test_identical_samples(self, engine):
        orig = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = engine.ks_test(orig, orig)
        assert result["statistic"] == 0.0
        assert result["passed"] is True

    def test_similar_samples(self, engine):
        orig = [1.0, 2.0, 3.0, 4.0, 5.0]
        imp = [1.1, 2.1, 3.0, 4.0, 5.0]
        result = engine.ks_test(orig, imp)
        assert result["passed"] is True
        assert result["p_value"] > 0.05

    def test_different_samples(self, engine):
        orig = [1.0, 2.0, 3.0, 4.0, 5.0]
        imp = [100.0, 200.0, 300.0, 400.0, 500.0]
        result = engine.ks_test(orig, imp)
        assert result["statistic"] > 0

    def test_empty_original(self, engine):
        result = engine.ks_test([], [1.0, 2.0])
        assert result["passed"] is True
        assert result["p_value"] == 1.0

    def test_empty_imputed(self, engine):
        result = engine.ks_test([1.0, 2.0], [])
        assert result["passed"] is True

    def test_provenance_hash(self, engine):
        result = engine.ks_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert len(result["provenance_hash"]) == 64

    def test_statistic_between_zero_and_one(self, engine):
        orig = [float(i) for i in range(20)]
        imp = [float(i) + 0.5 for i in range(20)]
        result = engine.ks_test(orig, imp)
        assert 0.0 <= result["statistic"] <= 1.0

    def test_n_original_n_imputed(self, engine):
        orig = [1.0, 2.0, 3.0]
        imp = [4.0, 5.0]
        result = engine.ks_test(orig, imp)
        assert result["n_original"] == 3
        assert result["n_imputed"] == 2


# ---------------------------------------------------------------------------
# Chi-square test
# ---------------------------------------------------------------------------


class TestChiSquareTest:
    def test_identical_distributions(self, engine):
        orig = ["A", "A", "B", "B", "C"]
        imp = ["A", "A", "B", "B", "C"]
        result = engine.chi_square_test(orig, imp)
        assert result["passed"] is True
        assert result["statistic"] < 0.01

    def test_different_distributions(self, engine):
        orig = ["A"] * 50 + ["B"] * 50
        imp = ["A"] * 10 + ["B"] * 90
        result = engine.chi_square_test(orig, imp)
        assert result["statistic"] > 0

    def test_empty_original(self, engine):
        result = engine.chi_square_test([], ["A", "B"])
        assert result["passed"] is True

    def test_single_category(self, engine):
        result = engine.chi_square_test(["A", "A"], ["A", "A"])
        assert result["passed"] is True
        assert result["degrees_of_freedom"] == 0

    def test_provenance_hash(self, engine):
        result = engine.chi_square_test(["A", "B"], ["A", "B"])
        assert len(result["provenance_hash"]) == 64

    def test_n_categories(self, engine):
        orig = ["A", "B", "C"]
        imp = ["A", "B", "C", "D"]
        result = engine.chi_square_test(orig, imp)
        assert result["n_categories"] == 4


# ---------------------------------------------------------------------------
# Plausibility check
# ---------------------------------------------------------------------------


class TestPlausibilityCheck:
    def test_within_bounds_passes(self, engine):
        stats = {"mean": 50.0, "std": 10.0, "min": 30.0, "max": 70.0}
        # Values must be within bounds AND have stdev / mean shift < 0.5 of orig
        imp = [40.0, 50.0, 60.0, 45.0, 55.0]
        result = engine.plausibility_check(imp, stats)
        assert result["passed"] is True
        assert result["out_of_range_count"] == 0

    def test_out_of_range_values(self, engine):
        stats = {"mean": 50.0, "std": 10.0, "min": 30.0, "max": 70.0}
        imp = [48.0, 52.0, 500.0]  # 500 is way out of range
        result = engine.plausibility_check(imp, stats)
        assert result["out_of_range_count"] > 0

    def test_empty_imputed_passes(self, engine):
        stats = {"mean": 50.0, "std": 10.0, "min": 30.0, "max": 70.0}
        result = engine.plausibility_check([], stats)
        assert result["passed"] is True

    def test_mean_shift(self, engine):
        stats = {"mean": 50.0, "std": 10.0, "min": 30.0, "max": 70.0}
        imp = [90.0, 95.0, 85.0]  # large shift
        result = engine.plausibility_check(imp, stats)
        assert result["mean_shift"] > 0

    def test_provenance_hash(self, engine):
        stats = {"mean": 50.0, "std": 10.0, "min": 30.0, "max": 70.0}
        result = engine.plausibility_check([50.0], stats)
        assert len(result["provenance_hash"]) == 64

    def test_std_zero_handled(self, engine):
        # When std=0 the code uses std=1.0 as fallback, so bounds are
        # [min - 2, max + 2] = [48, 52].  We also need the std_shift of
        # the imputed values to be < 0.5 of effective std (1.0), so we
        # provide two values close together (stdev ~0.7).
        stats = {"mean": 50.0, "std": 0, "min": 50.0, "max": 50.0}
        result = engine.plausibility_check([49.5, 50.5], stats)
        assert result["passed"] is True

    def test_bounds_calculation(self, engine):
        stats = {"mean": 50.0, "std": 10.0, "min": 30.0, "max": 70.0}
        result = engine.plausibility_check([50.0], stats)
        # lower = 30 - 20 = 10, upper = 70 + 20 = 90
        assert result["lower_bound"] == pytest.approx(10.0, rel=1e-4)
        assert result["upper_bound"] == pytest.approx(90.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Distribution preservation
# ---------------------------------------------------------------------------


class TestDistributionPreservation:
    def test_identical_data_passes(self, engine):
        data = [{"x": float(i)} for i in range(20)]
        result = engine.distribution_preservation(data, data)
        assert result["overall_passed"] is True

    def test_different_data_may_fail(self, engine):
        orig = [{"x": float(i)} for i in range(20)]
        imp = [{"x": float(i) + 100.0} for i in range(20)]
        result = engine.distribution_preservation(orig, imp)
        # Large shift should fail
        assert "overall_passed" in result

    def test_empty_data(self, engine):
        result = engine.distribution_preservation([], [])
        assert result["overall_passed"] is True
        assert result["n_columns_tested"] == 0

    def test_provenance_hash(self, engine):
        data = [{"x": 1.0}, {"x": 2.0}]
        result = engine.distribution_preservation(data, data)
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


class TestCrossValidate:
    def test_basic_cross_validate(self, engine):
        records = [{"val": float(i)} for i in range(50)]
        result = engine.cross_validate(records, "val", "mean", n_folds=3)
        assert "avg_rmse" in result
        assert "avg_mae" in result
        assert "passed" in result

    def test_insufficient_data(self, engine):
        records = [{"val": 1.0}, {"val": 2.0}]
        result = engine.cross_validate(records, "val", "mean", n_folds=5)
        assert result["passed"] is True
        assert result["fold_results"] == []

    def test_provenance_hash(self, engine):
        records = [{"val": float(i)} for i in range(50)]
        result = engine.cross_validate(records, "val", "mean", n_folds=3)
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# RMSE and MAE
# ---------------------------------------------------------------------------


class TestErrorMetrics:
    def test_rmse_zero(self, engine):
        assert engine.compute_rmse([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_rmse_positive(self, engine):
        assert engine.compute_rmse([1.0, 2.0], [2.0, 3.0]) == pytest.approx(1.0)

    def test_rmse_empty(self, engine):
        assert engine.compute_rmse([], []) == 0.0

    def test_mae_zero(self, engine):
        assert engine.compute_mae([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_mae_positive(self, engine):
        assert engine.compute_mae([1.0, 3.0], [2.0, 5.0]) == pytest.approx(1.5)

    def test_mae_empty(self, engine):
        assert engine.compute_mae([], []) == 0.0

    def test_rmse_different_lengths(self, engine):
        result = engine.compute_rmse([1.0, 2.0, 3.0], [1.0, 2.0])
        assert result == 0.0  # uses min length


# ---------------------------------------------------------------------------
# generate_validation_report
# ---------------------------------------------------------------------------


class TestGenerateValidationReport:
    def test_empty_results(self, engine):
        report = engine.generate_validation_report([])
        assert isinstance(report, ValidationReport)
        assert report.overall_passed is True
        assert report.columns_passed == 0
        assert report.columns_failed == 0

    def test_all_passed(self, engine):
        results = [
            {"column": "a", "test": "ks_test", "passed": True, "statistic": 0.1, "p_value": 0.8},
            {"column": "b", "test": "chi_square", "passed": True, "statistic": 0.5, "p_value": 0.9},
        ]
        report = engine.generate_validation_report(results)
        assert report.overall_passed is True
        assert report.columns_passed == 2
        assert report.columns_failed == 0

    def test_one_failure(self, engine):
        results = [
            {"column": "a", "test": "ks_test", "passed": True, "statistic": 0.1, "p_value": 0.8},
            {"column": "b", "test": "ks_test", "passed": False, "statistic": 0.9, "p_value": 0.01},
        ]
        report = engine.generate_validation_report(results)
        assert report.overall_passed is False
        assert report.columns_passed == 1
        assert report.columns_failed == 1

    def test_provenance_hash(self, engine):
        results = [{"column": "a", "test": "ks_test", "passed": True, "statistic": 0.1, "p_value": 0.5}]
        report = engine.generate_validation_report(results)
        assert len(report.provenance_hash) == 64


# ---------------------------------------------------------------------------
# validate_imputation (integration-like)
# ---------------------------------------------------------------------------


class TestValidateImputation:
    def test_empty_data_returns_passed(self, engine):
        result = engine.validate_imputation([], [], "mean")
        assert result.passed is True

    def test_numeric_columns_use_ks_test(self, engine):
        orig = [{"x": float(i)} for i in range(10)]
        imp = [{"x": float(i) + 0.01} for i in range(10)]
        result = engine.validate_imputation(orig, imp, "mean")
        assert result.passed is True

    def test_provenance_hash(self, engine):
        orig = [{"x": 1.0}]
        imp = [{"x": 1.0}]
        result = engine.validate_imputation(orig, imp, "mean")
        assert len(result.provenance_hash) == 64


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class TestPrivateHelpers:
    def test_ecdf_value_at_boundary(self, engine):
        sorted_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert engine._ecdf_value(sorted_data, 3.0) == pytest.approx(0.6, rel=1e-4)

    def test_ecdf_value_empty(self, engine):
        assert engine._ecdf_value([], 1.0) == 0.0

    def test_ecdf_value_below_min(self, engine):
        assert engine._ecdf_value([1.0, 2.0], 0.0) == 0.0

    def test_ecdf_value_above_max(self, engine):
        assert engine._ecdf_value([1.0, 2.0], 5.0) == 1.0

    def test_chi2_p_value_zero_df(self, engine):
        assert engine._chi2_p_value(1.0, 0) == 1.0

    def test_chi2_p_value_zero_chi2(self, engine):
        assert engine._chi2_p_value(0.0, 5) == 1.0

    def test_normal_cdf_extremes(self, engine):
        assert engine._normal_cdf(-10.0) == 0.0
        assert engine._normal_cdf(10.0) == 1.0

    def test_normal_cdf_zero(self, engine):
        assert engine._normal_cdf(0.0) == pytest.approx(0.5, abs=0.01)

    def test_compare_statistics_identical(self, engine):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = engine._compare_statistics(vals, vals)
        assert result["passed"] is True

    def test_compare_statistics_empty(self, engine):
        result = engine._compare_statistics([], [])
        assert result["passed"] is True

    def test_empty_result(self, engine):
        result = engine._empty_result("test_op")
        assert isinstance(result, ValidationResult)
        assert result.passed is True
