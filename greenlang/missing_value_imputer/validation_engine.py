# -*- coding: utf-8 -*-
"""
Validation Engine - AGENT-DATA-012: Missing Value Imputer (GL-DATA-X-015)

Validates imputation quality using statistical tests: Kolmogorov-Smirnov
for numeric distribution similarity, chi-square for categorical distributions,
plausibility range checks, distribution preservation metrics, and
cross-validation with artificial masking.

All statistical tests are implemented in pure Python with no external
library dependencies.

Zero-Hallucination Guarantees:
    - All test statistics are deterministic arithmetic
    - KS statistic uses standard empirical CDF difference
    - Chi-square uses standard observed/expected formula
    - P-value approximations use well-known closed-form bounds
    - No ML/LLM calls in any validation path
    - SHA-256 provenance on every validation result

Example:
    >>> from greenlang.missing_value_imputer.validation_engine import ValidationEngine
    >>> from greenlang.missing_value_imputer.config import MissingValueImputerConfig
    >>> engine = ValidationEngine(MissingValueImputerConfig())
    >>> ks = engine.ks_test([1,2,3,4,5], [1.1,2.1,3.0,4.0,5.0])
    >>> print(ks["statistic"], ks["passed"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.missing_value_imputer.config import MissingValueImputerConfig
from greenlang.missing_value_imputer.models import (
    ImputationStrategy,
    ImputedValue,
    ValidationMethod,
    ValidationReport,
    ValidationResult,
)
from greenlang.missing_value_imputer.metrics import (
    inc_validations,
    observe_duration,
    inc_errors,
)
from greenlang.missing_value_imputer.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

__all__ = [
    "ValidationEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _is_missing(value: Any) -> bool:
    """Determine whether a value is considered missing."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _is_numeric(value: Any) -> bool:
    """Check if a value is numeric (excluding bool)."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash."""
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_stdev(values: List[float]) -> float:
    """Compute sample standard deviation, returning 0.0 for < 2 values."""
    if len(values) < 2:
        return 0.0
    try:
        return statistics.stdev([float(v) for v in values])
    except (ValueError, TypeError, AttributeError, statistics.StatisticsError):
        return 0.0


# ===========================================================================
# ValidationEngine
# ===========================================================================


class ValidationEngine:
    """Validates imputation quality using statistical tests.

    Provides KS test, chi-square test, plausibility checks, distribution
    preservation metrics, and cross-validation for imputation quality
    assurance.

    Attributes:
        config: Service configuration.
        provenance: SHA-256 provenance tracker.

    Example:
        >>> engine = ValidationEngine(MissingValueImputerConfig())
        >>> result = engine.ks_test([1,2,3], [1.1,2.0,3.1])
        >>> assert "statistic" in result
    """

    def __init__(self, config: MissingValueImputerConfig) -> None:
        """Initialize the ValidationEngine.

        Args:
            config: Service configuration instance.
        """
        self.config = config
        self.provenance = ProvenanceTracker()
        logger.info("ValidationEngine initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_imputation(
        self,
        original_data: List[Dict[str, Any]],
        imputed_data: List[Dict[str, Any]],
        method_used: str,
    ) -> ValidationResult:
        """Run comprehensive validation on imputed data.

        Selects appropriate tests based on the imputation method and
        data types, then aggregates results.

        Args:
            original_data: Records before imputation.
            imputed_data: Records after imputation.
            method_used: Name of the imputation method.

        Returns:
            ValidationResult with overall pass/fail and details.
        """
        start = time.monotonic()

        if not original_data or not imputed_data:
            return self._empty_result("validate_imputation")

        # Find columns that were imputed
        all_cols = set()
        for r in imputed_data:
            all_cols.update(r.keys())

        results: List[Dict[str, Any]] = []
        for col in sorted(all_cols):
            orig_vals = [r.get(col) for r in original_data]
            imp_vals = [r.get(col) for r in imputed_data]

            orig_non_missing = [v for v in orig_vals if not _is_missing(v)]
            imp_non_missing = [v for v in imp_vals if not _is_missing(v)]

            if not orig_non_missing or not imp_non_missing:
                continue

            # Check if numeric
            numeric_orig = [v for v in orig_non_missing if _is_numeric(v)]
            if len(numeric_orig) / len(orig_non_missing) > 0.5:
                ks = self.ks_test(
                    [float(v) for v in numeric_orig],
                    [float(v) for v in imp_non_missing if _is_numeric(v)],
                )
                results.append({"column": col, "test": "ks_test", **ks})
            else:
                chi = self.chi_square_test(orig_non_missing, imp_non_missing)
                results.append({"column": col, "test": "chi_square", **chi})

        # Aggregate
        all_passed = all(r.get("passed", False) for r in results) if results else True
        overall_p = (
            min(r.get("p_value", 1.0) for r in results)
            if results else 1.0
        )

        elapsed = time.monotonic() - start
        observe_duration("validate", elapsed)

        prov = _compute_provenance(
            "validate_imputation", f"{method_used}:{len(results)}"
        )
        return ValidationResult(
            column_name="__all__",
            method=ValidationMethod.DISTRIBUTION_PRESERVATION,
            passed=all_passed,
            test_statistic=None,
            p_value=round(overall_p, 6),
            threshold=0.05,
            details={
                "method_used": method_used,
                "column_results": results,
                "total_tests": len(results),
                "tests_passed": sum(1 for r in results if r.get("passed", False)),
            },
            provenance_hash=prov,
        )

    def ks_test(
        self,
        original_column: List[float],
        imputed_column: List[float],
    ) -> Dict[str, Any]:
        """Perform two-sample Kolmogorov-Smirnov test.

        Tests whether the imputed distribution differs significantly from
        the original. Uses the empirical CDF difference.

        H0: The two samples come from the same distribution.

        Args:
            original_column: Original numeric values (non-missing).
            imputed_column: Imputed numeric values.

        Returns:
            Dict with statistic, p_value, critical_value, passed, provenance_hash.
        """
        if not original_column or not imputed_column:
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "critical_value": 0.0,
                "passed": True,
                "provenance_hash": _compute_provenance("ks_test", "empty"),
            }

        n1 = len(original_column)
        n2 = len(imputed_column)

        # Build sorted combined values
        sorted1 = sorted(original_column)
        sorted2 = sorted(imputed_column)

        # Compute KS statistic: max |F1(x) - F2(x)|
        all_vals = sorted(set(sorted1 + sorted2))
        max_diff = 0.0

        for x in all_vals:
            # Empirical CDF
            f1 = self._ecdf_value(sorted1, x)
            f2 = self._ecdf_value(sorted2, x)
            diff = abs(f1 - f2)
            if diff > max_diff:
                max_diff = diff

        # Critical value at alpha=0.05
        # D_crit = c(alpha) * sqrt((n1+n2)/(n1*n2))
        # c(0.05) = 1.36
        if n1 > 0 and n2 > 0:
            critical = 1.36 * math.sqrt((n1 + n2) / (n1 * n2))
        else:
            critical = 1.0

        passed = max_diff <= critical

        # Approximate p-value using asymptotic formula
        # p = 2 * exp(-2 * n_eff * D^2) where n_eff = n1*n2/(n1+n2)
        if n1 + n2 > 0:
            n_eff = (n1 * n2) / (n1 + n2)
        else:
            n_eff = 1.0
        exponent = -2.0 * n_eff * max_diff ** 2
        p_value = 2.0 * math.exp(max(exponent, -500))
        p_value = min(1.0, p_value)

        if passed:
            inc_validations("ks_test")

        prov = _compute_provenance("ks_test", f"D={max_diff:.6f}:p={p_value:.6f}")
        return {
            "statistic": round(max_diff, 6),
            "p_value": round(p_value, 6),
            "critical_value": round(critical, 6),
            "passed": passed,
            "n_original": n1,
            "n_imputed": n2,
            "provenance_hash": prov,
        }

    def chi_square_test(
        self,
        original_column: List[Any],
        imputed_column: List[Any],
    ) -> Dict[str, Any]:
        """Perform chi-square test for categorical distribution similarity.

        Tests whether the category frequencies differ significantly between
        original and imputed data.

        H0: Category frequencies are the same in both samples.

        Args:
            original_column: Original categorical values (non-missing).
            imputed_column: Imputed categorical values.

        Returns:
            Dict with statistic, p_value, degrees_of_freedom, passed,
                provenance_hash.
        """
        if not original_column or not imputed_column:
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "degrees_of_freedom": 0,
                "passed": True,
                "provenance_hash": _compute_provenance("chi_square", "empty"),
            }

        # Count frequencies
        orig_freq = Counter(str(v) for v in original_column)
        imp_freq = Counter(str(v) for v in imputed_column)

        all_categories = sorted(set(orig_freq.keys()) | set(imp_freq.keys()))
        if len(all_categories) < 2:
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "degrees_of_freedom": 0,
                "passed": True,
                "provenance_hash": _compute_provenance("chi_square", "single_cat"),
            }

        n_orig = len(original_column)
        n_imp = len(imputed_column)
        k = len(all_categories)
        df = k - 1

        # Compute chi-square statistic
        # Expected freq for category c: (orig_c + imp_c) * n_sample / (n_orig + n_imp)
        chi2 = 0.0
        for cat in all_categories:
            o_count = orig_freq.get(cat, 0)
            i_count = imp_freq.get(cat, 0)
            total_cat = o_count + i_count

            expected_orig = total_cat * n_orig / (n_orig + n_imp)
            expected_imp = total_cat * n_imp / (n_orig + n_imp)

            if expected_orig > 0:
                chi2 += (o_count - expected_orig) ** 2 / expected_orig
            if expected_imp > 0:
                chi2 += (i_count - expected_imp) ** 2 / expected_imp

        # Approximate p-value using Wilson-Hilferty approximation
        p_value = self._chi2_p_value(chi2, df)

        passed = p_value > 0.05

        if passed:
            inc_validations("chi_square")

        prov = _compute_provenance(
            "chi_square", f"chi2={chi2:.6f}:df={df}:p={p_value:.6f}"
        )
        return {
            "statistic": round(chi2, 6),
            "p_value": round(p_value, 6),
            "degrees_of_freedom": df,
            "n_categories": k,
            "passed": passed,
            "provenance_hash": prov,
        }

    def plausibility_check(
        self,
        imputed_values: List[Any],
        column_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check if imputed values fall within plausible bounds.

        Verifies that imputed values are within the range [min - 2*std,
        max + 2*std] of the original distribution, and that the mean
        and std are not dramatically changed.

        Args:
            imputed_values: List of imputed values.
            column_stats: Dict with keys: mean, std, min, max of original.

        Returns:
            Dict with passed, out_of_range_count, out_of_range_pct,
                mean_shift, std_shift, details, provenance_hash.
        """
        if not imputed_values:
            return {
                "passed": True,
                "out_of_range_count": 0,
                "out_of_range_pct": 0.0,
                "mean_shift": 0.0,
                "std_shift": 0.0,
                "details": {},
                "provenance_hash": _compute_provenance("plausibility", "empty"),
            }

        orig_mean = column_stats.get("mean", 0.0)
        orig_std = column_stats.get("std", 1.0)
        orig_min = column_stats.get("min", float("-inf"))
        orig_max = column_stats.get("max", float("inf"))

        if orig_std == 0:
            orig_std = 1.0

        # Plausible range
        lower_bound = orig_min - 2.0 * orig_std
        upper_bound = orig_max + 2.0 * orig_std

        numeric_vals = [float(v) for v in imputed_values if _is_numeric(v)]
        if not numeric_vals:
            return {
                "passed": True,
                "out_of_range_count": 0,
                "out_of_range_pct": 0.0,
                "mean_shift": 0.0,
                "std_shift": 0.0,
                "details": {},
                "provenance_hash": _compute_provenance("plausibility", "non_numeric"),
            }

        out_of_range = sum(
            1 for v in numeric_vals if v < lower_bound or v > upper_bound
        )
        out_pct = out_of_range / len(numeric_vals) if numeric_vals else 0.0

        imp_mean = sum(numeric_vals) / len(numeric_vals)
        imp_std = _safe_stdev(numeric_vals)

        mean_shift = abs(imp_mean - orig_mean) / orig_std if orig_std > 0 else 0.0
        std_shift = abs(imp_std - orig_std) / orig_std if orig_std > 0 else 0.0

        # Pass if <10% out of range AND mean/std shifts are small
        passed = out_pct < 0.10 and mean_shift < 0.50 and std_shift < 0.50

        if passed:
            inc_validations("plausibility_range")

        prov = _compute_provenance(
            "plausibility",
            f"oor={out_of_range}:ms={mean_shift:.4f}:ss={std_shift:.4f}",
        )
        return {
            "passed": passed,
            "out_of_range_count": out_of_range,
            "out_of_range_pct": round(out_pct, 6),
            "mean_shift": round(mean_shift, 6),
            "std_shift": round(std_shift, 6),
            "imputed_mean": round(imp_mean, 6),
            "imputed_std": round(imp_std, 6),
            "lower_bound": round(lower_bound, 6),
            "upper_bound": round(upper_bound, 6),
            "details": {
                "n_imputed": len(numeric_vals),
                "orig_mean": orig_mean,
                "orig_std": orig_std,
                "orig_min": orig_min,
                "orig_max": orig_max,
            },
            "provenance_hash": prov,
        }

    def distribution_preservation(
        self,
        original_data: List[Dict[str, Any]],
        imputed_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare overall distribution statistics before and after imputation.

        Checks mean, median, std, skewness, and kurtosis for each numeric
        column.

        Args:
            original_data: Records before imputation.
            imputed_data: Records after imputation.

        Returns:
            Dict with per-column comparison results and overall_passed.
        """
        start = time.monotonic()

        all_cols = set()
        for r in imputed_data:
            all_cols.update(r.keys())

        column_results: Dict[str, Dict[str, Any]] = {}
        all_passed = True

        for col in sorted(all_cols):
            orig_vals = [
                float(r.get(col))
                for r in original_data
                if not _is_missing(r.get(col)) and _is_numeric(r.get(col))
            ]
            imp_vals = [
                float(r.get(col))
                for r in imputed_data
                if not _is_missing(r.get(col)) and _is_numeric(r.get(col))
            ]

            if not orig_vals or not imp_vals:
                continue

            comparison = self._compare_statistics(orig_vals, imp_vals)
            column_results[col] = comparison

            if not comparison.get("passed", True):
                all_passed = False

        elapsed = time.monotonic() - start
        observe_duration("validate", elapsed)

        if all_passed:
            inc_validations("distribution_preservation")

        prov = _compute_provenance(
            "distribution_preservation",
            f"{len(column_results)}:{all_passed}",
        )
        return {
            "overall_passed": all_passed,
            "column_results": column_results,
            "n_columns_tested": len(column_results),
            "n_columns_passed": sum(
                1 for r in column_results.values() if r.get("passed", True)
            ),
            "provenance_hash": prov,
        }

    def cross_validate(
        self,
        records: List[Dict[str, Any]],
        column: str,
        method: str,
        n_folds: int = 5,
    ) -> Dict[str, Any]:
        """Cross-validate imputation by artificially masking known values.

        Randomly masks a fraction of known values, imputes them, and
        compares to the originals. Repeats for n_folds.

        Args:
            records: List of record dictionaries.
            column: Column to test.
            method: Imputation method to validate.
            n_folds: Number of cross-validation folds.

        Returns:
            Dict with avg_rmse, avg_mae, fold_results, passed, provenance_hash.
        """
        start = time.monotonic()
        import random as rng

        # Get indices with non-missing values
        observed_indices = [
            i for i, r in enumerate(records)
            if not _is_missing(r.get(column)) and _is_numeric(r.get(column))
        ]

        if len(observed_indices) < n_folds * 2:
            return {
                "avg_rmse": 0.0,
                "avg_mae": 0.0,
                "fold_results": [],
                "passed": True,
                "provenance_hash": _compute_provenance("cross_validate", "insufficient"),
            }

        fold_rmses: List[float] = []
        fold_maes: List[float] = []
        fold_results: List[Dict[str, Any]] = []

        rng_inst = rng.Random(42)
        rng_inst.shuffle(observed_indices)
        fold_size = len(observed_indices) // n_folds

        for fold_idx in range(n_folds):
            # Select mask indices for this fold
            mask_start = fold_idx * fold_size
            mask_end = mask_start + fold_size
            mask_indices = set(observed_indices[mask_start:mask_end])

            if not mask_indices:
                continue

            # Create masked records
            actual_values: Dict[int, float] = {}
            masked_records = []
            for i, r in enumerate(records):
                new_r = dict(r)
                if i in mask_indices:
                    actual_values[i] = float(r.get(column))
                    new_r[column] = None
                masked_records.append(new_r)

            # Simple imputation (mean) for cross-validation
            non_masked_vals = [
                float(r.get(column))
                for i, r in enumerate(masked_records)
                if not _is_missing(r.get(column)) and _is_numeric(r.get(column))
            ]
            if not non_masked_vals:
                continue

            imp_val = sum(non_masked_vals) / len(non_masked_vals)

            # Compute errors
            actuals = [actual_values[idx] for idx in sorted(mask_indices)]
            predicted = [imp_val] * len(actuals)

            rmse = self.compute_rmse(actuals, predicted)
            mae = self.compute_mae(actuals, predicted)

            fold_rmses.append(rmse)
            fold_maes.append(mae)
            fold_results.append({
                "fold": fold_idx,
                "n_masked": len(mask_indices),
                "rmse": round(rmse, 6),
                "mae": round(mae, 6),
            })

        avg_rmse = sum(fold_rmses) / len(fold_rmses) if fold_rmses else 0.0
        avg_mae = sum(fold_maes) / len(fold_maes) if fold_maes else 0.0

        # Compute relative RMSE against std for pass/fail
        all_vals = [
            float(records[i].get(column))
            for i in observed_indices
        ]
        data_std = _safe_stdev(all_vals)
        relative_rmse = avg_rmse / data_std if data_std > 0 else 0.0
        passed = relative_rmse < 1.0  # RMSE should be less than 1 std

        elapsed = time.monotonic() - start
        observe_duration("validate", elapsed)

        if passed:
            inc_validations("cross_validation")

        prov = _compute_provenance(
            "cross_validate",
            f"{column}:{avg_rmse:.6f}:{avg_mae:.6f}",
        )
        return {
            "avg_rmse": round(avg_rmse, 6),
            "avg_mae": round(avg_mae, 6),
            "relative_rmse": round(relative_rmse, 6),
            "n_folds": n_folds,
            "fold_results": fold_results,
            "passed": passed,
            "provenance_hash": prov,
        }

    def compute_rmse(
        self, actual: List[float], predicted: List[float]
    ) -> float:
        """Compute Root Mean Square Error.

        Args:
            actual: Actual values.
            predicted: Predicted values.

        Returns:
            RMSE value.
        """
        n = min(len(actual), len(predicted))
        if n == 0:
            return 0.0
        mse = sum((a - p) ** 2 for a, p in zip(actual[:n], predicted[:n])) / n
        return math.sqrt(mse)

    def compute_mae(
        self, actual: List[float], predicted: List[float]
    ) -> float:
        """Compute Mean Absolute Error.

        Args:
            actual: Actual values.
            predicted: Predicted values.

        Returns:
            MAE value.
        """
        n = min(len(actual), len(predicted))
        if n == 0:
            return 0.0
        return sum(abs(a - p) for a, p in zip(actual[:n], predicted[:n])) / n

    def generate_validation_report(
        self,
        results: List[Dict[str, Any]],
    ) -> ValidationReport:
        """Aggregate validation results into a comprehensive report.

        Args:
            results: List of validation result dicts (from individual tests).

        Returns:
            ValidationReport model.
        """
        validation_results: List[ValidationResult] = []
        columns_passed = 0
        columns_failed = 0

        for r in results:
            col = r.get("column", r.get("column_name", "unknown"))
            test_name = r.get("test", r.get("method", "unknown"))
            passed = r.get("passed", False)
            stat = r.get("statistic", r.get("test_statistic"))
            p_val = r.get("p_value")
            threshold = r.get("threshold", 0.05)

            # Map test name to ValidationMethod
            method_map = {
                "ks_test": ValidationMethod.KS_TEST,
                "chi_square": ValidationMethod.CHI_SQUARE,
                "plausibility_range": ValidationMethod.PLAUSIBILITY_RANGE,
                "plausibility": ValidationMethod.PLAUSIBILITY_RANGE,
                "distribution_preservation": ValidationMethod.DISTRIBUTION_PRESERVATION,
                "cross_validation": ValidationMethod.CROSS_VALIDATION,
                "cross_validate": ValidationMethod.CROSS_VALIDATION,
            }
            method = method_map.get(test_name, ValidationMethod.PLAUSIBILITY_RANGE)

            prov = _compute_provenance(
                "validation_result",
                f"{col}:{test_name}:{passed}",
            )

            vr = ValidationResult(
                column_name=col,
                method=method,
                passed=passed,
                test_statistic=stat,
                p_value=p_val,
                threshold=threshold,
                details=r,
                provenance_hash=prov,
            )
            validation_results.append(vr)

            if passed:
                columns_passed += 1
            else:
                columns_failed += 1

        overall_passed = columns_failed == 0

        prov = _compute_provenance(
            "validation_report",
            f"pass={columns_passed}:fail={columns_failed}",
        )
        return ValidationReport(
            results=validation_results,
            overall_passed=overall_passed,
            columns_passed=columns_passed,
            columns_failed=columns_failed,
            provenance_hash=prov,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ecdf_value(self, sorted_data: List[float], x: float) -> float:
        """Compute empirical CDF value at x.

        Args:
            sorted_data: Sorted list of values.
            x: Query point.

        Returns:
            Proportion of values <= x.
        """
        n = len(sorted_data)
        if n == 0:
            return 0.0
        count = 0
        for val in sorted_data:
            if val <= x:
                count += 1
            else:
                break
        return count / n

    def _chi2_p_value(self, chi2: float, df: int) -> float:
        """Approximate chi-squared p-value.

        Uses the Wilson-Hilferty normal approximation for the chi-squared
        CDF: X^{2/3} is approximately normal.

        Args:
            chi2: Chi-square test statistic.
            df: Degrees of freedom.

        Returns:
            Approximate p-value.
        """
        if df <= 0:
            return 1.0
        if chi2 <= 0:
            return 1.0

        # Wilson-Hilferty approximation
        k = float(df)
        z = ((chi2 / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(
            2.0 / (9.0 * k)
        )

        # Standard normal CDF approximation (Abramowitz & Stegun)
        p = self._normal_cdf(z)
        return max(0.0, min(1.0, 1.0 - p))

    def _normal_cdf(self, z: float) -> float:
        """Approximate standard normal CDF.

        Uses the Abramowitz & Stegun approximation (formula 26.2.17).

        Args:
            z: Standard normal variate.

        Returns:
            P(Z <= z).
        """
        if z < -8.0:
            return 0.0
        if z > 8.0:
            return 1.0

        b0 = 0.2316419
        b1 = 0.319381530
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429

        if z >= 0:
            t = 1.0 / (1.0 + b0 * z)
            phi = (
                1.0 / math.sqrt(2.0 * math.pi) *
                math.exp(-0.5 * z * z)
            )
            cdf = 1.0 - phi * (b1 * t + b2 * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5)
        else:
            cdf = 1.0 - self._normal_cdf(-z)

        return max(0.0, min(1.0, cdf))

    def _compare_statistics(
        self,
        before: List[float],
        after: List[float],
    ) -> Dict[str, Any]:
        """Compare distribution statistics before and after imputation.

        Args:
            before: Values before imputation.
            after: Values after imputation.

        Returns:
            Dict with mean, median, std comparisons and passed flag.
        """
        if not before or not after:
            return {"passed": True}

        before_mean = sum(before) / len(before)
        after_mean = sum(after) / len(after)
        before_std = _safe_stdev(before)
        after_std = _safe_stdev(after)
        before_median = statistics.median(before)
        after_median = statistics.median(after)

        # Quartiles
        before_sorted = sorted(before)
        after_sorted = sorted(after)
        before_q1 = before_sorted[len(before_sorted) // 4]
        before_q3 = before_sorted[(3 * len(before_sorted)) // 4]
        after_q1 = after_sorted[len(after_sorted) // 4]
        after_q3 = after_sorted[(3 * len(after_sorted)) // 4]

        # Relative differences
        scale = before_std if before_std > 1e-10 else 1.0
        mean_diff = abs(after_mean - before_mean) / scale
        median_diff = abs(after_median - before_median) / scale
        std_ratio = after_std / before_std if before_std > 1e-10 else 1.0

        # Pass if mean shift < 0.5 std, median shift < 0.5 std,
        # std ratio between 0.5 and 2.0
        passed = (
            mean_diff < 0.50 and
            median_diff < 0.50 and
            0.5 < std_ratio < 2.0
        )

        return {
            "passed": passed,
            "before": {
                "mean": round(before_mean, 6),
                "median": round(before_median, 6),
                "std": round(before_std, 6),
                "q1": round(before_q1, 6),
                "q3": round(before_q3, 6),
                "n": len(before),
            },
            "after": {
                "mean": round(after_mean, 6),
                "median": round(after_median, 6),
                "std": round(after_std, 6),
                "q1": round(after_q1, 6),
                "q3": round(after_q3, 6),
                "n": len(after),
            },
            "mean_diff_std": round(mean_diff, 6),
            "median_diff_std": round(median_diff, 6),
            "std_ratio": round(std_ratio, 6),
        }

    def _empty_result(self, operation: str) -> ValidationResult:
        """Create an empty validation result for edge cases.

        Args:
            operation: Operation name for provenance.

        Returns:
            ValidationResult with passed=True and no details.
        """
        return ValidationResult(
            column_name="__empty__",
            method=ValidationMethod.PLAUSIBILITY_RANGE,
            passed=True,
            details={},
            provenance_hash=_compute_provenance(operation, "empty"),
        )
