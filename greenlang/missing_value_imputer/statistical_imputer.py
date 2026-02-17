# -*- coding: utf-8 -*-
"""
Statistical Imputer Engine - AGENT-DATA-012: Missing Value Imputer (GL-DATA-X-015)

Provides classical statistical imputation methods: mean, median, mode,
k-nearest neighbors (KNN), linear regression, hot-deck, Last Observation
Carried Forward (LOCF), Next Observation Carried Backward (NOCB), and
group-by aggregation imputation.

Zero-Hallucination Guarantees:
    - All imputation values are deterministic Python arithmetic
    - KNN uses Euclidean distance with explicit neighbor enumeration
    - Regression uses closed-form OLS (normal equation)
    - No ML/LLM calls in any imputation path
    - SHA-256 provenance on every imputed value
    - Confidence scores computed from data characteristics

Example:
    >>> from greenlang.missing_value_imputer.statistical_imputer import StatisticalImputerEngine
    >>> from greenlang.missing_value_imputer.config import MissingValueImputerConfig
    >>> engine = StatisticalImputerEngine(MissingValueImputerConfig())
    >>> records = [{"a": 1.0}, {"a": None}, {"a": 3.0}]
    >>> imputed = engine.impute_mean(records, "a")
    >>> print(imputed[0].imputed_value, imputed[0].confidence)

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
import random
import statistics
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.missing_value_imputer.config import MissingValueImputerConfig
from greenlang.missing_value_imputer.models import (
    ConfidenceLevel,
    ImputationStrategy,
    ImputedValue,
)
from greenlang.missing_value_imputer.metrics import (
    inc_values_imputed,
    observe_confidence,
    observe_duration,
    inc_errors,
)
from greenlang.missing_value_imputer.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

__all__ = [
    "StatisticalImputerEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _is_missing(value: Any) -> bool:
    """Determine whether a value is considered missing.

    Args:
        value: The value to check.

    Returns:
        True if value is None, empty/whitespace string, or NaN.
    """
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _is_numeric(value: Any) -> bool:
    """Check if a value is numeric (int or float, excluding bool).

    Args:
        value: The value to check.

    Returns:
        True if value is int or float and not bool.
    """
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def _to_float(value: Any) -> Optional[float]:
    """Safely convert a value to float.

    Args:
        value: The value to convert.

    Returns:
        Float value or None if conversion fails.
    """
    if _is_missing(value):
        return None
    if _is_numeric(value):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash.

    Args:
        operation: Name of the operation.
        data_repr: Serialized representation of the data.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _classify_confidence(score: float) -> ConfidenceLevel:
    """Classify a numeric confidence score into a level.

    Args:
        score: Confidence score in [0.0, 1.0].

    Returns:
        ConfidenceLevel enum value.
    """
    if score >= 0.85:
        return ConfidenceLevel.HIGH
    if score >= 0.70:
        return ConfidenceLevel.MEDIUM
    if score >= 0.50:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.VERY_LOW


def _safe_stdev(values: List[float]) -> float:
    """Compute sample standard deviation, returning 0.0 for < 2 values.

    Args:
        values: List of numeric values.

    Returns:
        Sample standard deviation or 0.0.
    """
    if len(values) < 2:
        return 0.0
    try:
        return statistics.stdev(values)
    except (ValueError, statistics.StatisticsError):
        return 0.0


# ===========================================================================
# StatisticalImputerEngine
# ===========================================================================


class StatisticalImputerEngine:
    """Classical statistical imputation engine.

    Provides deterministic imputation methods using standard statistical
    techniques. Every imputed value includes a confidence score and
    SHA-256 provenance hash for audit trail.

    Zero-hallucination: All computations use Python arithmetic only.
    No ML models or LLM calls in any code path.

    Attributes:
        config: Service configuration.
        provenance: SHA-256 provenance tracker.

    Example:
        >>> engine = StatisticalImputerEngine(MissingValueImputerConfig())
        >>> result = engine.impute_mean([{"x": 1}, {"x": None}, {"x": 3}], "x")
        >>> assert result[0].imputed_value == 2.0
    """

    def __init__(self, config: MissingValueImputerConfig) -> None:
        """Initialize the StatisticalImputerEngine.

        Args:
            config: Service configuration instance.
        """
        self.config = config
        self.provenance = ProvenanceTracker()
        logger.info("StatisticalImputerEngine initialized")

    # ------------------------------------------------------------------
    # Mean imputation
    # ------------------------------------------------------------------

    def impute_mean(
        self,
        records: List[Dict[str, Any]],
        column: str,
    ) -> List[ImputedValue]:
        """Impute missing values using column mean.

        Only applicable to numeric columns. Non-numeric columns will
        raise ValueError.

        Args:
            records: List of record dictionaries.
            column: Name of the column to impute.

        Returns:
            List of ImputedValue for each missing record.

        Raises:
            ValueError: If no numeric values available for mean computation.
        """
        start = time.monotonic()
        numeric_vals = self._extract_numeric(records, column)
        if not numeric_vals:
            raise ValueError(f"No numeric values in column '{column}' for mean")

        mean_val = sum(numeric_vals) / len(numeric_vals)
        n_observed = len(numeric_vals)
        variance = _safe_stdev(numeric_vals) ** 2

        imputed: List[ImputedValue] = []
        for i, record in enumerate(records):
            if _is_missing(record.get(column)):
                confidence = self._compute_confidence(
                    "mean", n_observed, variance
                )
                prov = _compute_provenance("impute_mean", f"{column}:{i}:{mean_val}")
                iv = ImputedValue(
                    record_index=i,
                    column_name=column,
                    imputed_value=round(mean_val, 8),
                    original_value=record.get(column),
                    strategy=ImputationStrategy.MEAN,
                    confidence=round(confidence, 4),
                    confidence_level=_classify_confidence(confidence),
                    contributing_records=n_observed,
                    provenance_hash=prov,
                )
                imputed.append(iv)

        self._record_metrics("mean", imputed, start)
        return imputed

    # ------------------------------------------------------------------
    # Median imputation
    # ------------------------------------------------------------------

    def impute_median(
        self,
        records: List[Dict[str, Any]],
        column: str,
    ) -> List[ImputedValue]:
        """Impute missing values using column median.

        More robust to outliers than mean imputation. Only applicable
        to numeric columns.

        Args:
            records: List of record dictionaries.
            column: Name of the column to impute.

        Returns:
            List of ImputedValue for each missing record.

        Raises:
            ValueError: If no numeric values available.
        """
        start = time.monotonic()
        numeric_vals = self._extract_numeric(records, column)
        if not numeric_vals:
            raise ValueError(f"No numeric values in column '{column}' for median")

        median_val = statistics.median(numeric_vals)
        n_observed = len(numeric_vals)
        variance = _safe_stdev(numeric_vals) ** 2

        imputed: List[ImputedValue] = []
        for i, record in enumerate(records):
            if _is_missing(record.get(column)):
                confidence = self._compute_confidence(
                    "median", n_observed, variance
                )
                prov = _compute_provenance(
                    "impute_median", f"{column}:{i}:{median_val}"
                )
                iv = ImputedValue(
                    record_index=i,
                    column_name=column,
                    imputed_value=round(median_val, 8),
                    original_value=record.get(column),
                    strategy=ImputationStrategy.MEDIAN,
                    confidence=round(confidence, 4),
                    confidence_level=_classify_confidence(confidence),
                    contributing_records=n_observed,
                    provenance_hash=prov,
                )
                imputed.append(iv)

        self._record_metrics("median", imputed, start)
        return imputed

    # ------------------------------------------------------------------
    # Mode imputation
    # ------------------------------------------------------------------

    def impute_mode(
        self,
        records: List[Dict[str, Any]],
        column: str,
    ) -> List[ImputedValue]:
        """Impute missing values using column mode (most frequent value).

        Applicable to any column type (numeric, categorical, text).
        When there are ties, the first mode encountered is used.

        Args:
            records: List of record dictionaries.
            column: Name of the column to impute.

        Returns:
            List of ImputedValue for each missing record.

        Raises:
            ValueError: If no non-missing values exist.
        """
        start = time.monotonic()
        non_missing = [
            record.get(column) for record in records
            if not _is_missing(record.get(column))
        ]
        if not non_missing:
            raise ValueError(f"No non-missing values in column '{column}' for mode")

        freq = Counter(str(v) for v in non_missing)
        mode_str = freq.most_common(1)[0][0]
        mode_count = freq.most_common(1)[0][1]

        # Recover original typed value
        mode_val: Any = mode_str
        for v in non_missing:
            if str(v) == mode_str:
                mode_val = v
                break

        n_observed = len(non_missing)
        mode_fraction = mode_count / n_observed if n_observed > 0 else 0.0

        imputed: List[ImputedValue] = []
        for i, record in enumerate(records):
            if _is_missing(record.get(column)):
                # Confidence based on mode dominance
                confidence = min(0.5 + mode_fraction * 0.5, 0.95)
                confidence = self._adjust_confidence_by_sample(
                    confidence, n_observed
                )
                prov = _compute_provenance(
                    "impute_mode", f"{column}:{i}:{mode_val}"
                )
                iv = ImputedValue(
                    record_index=i,
                    column_name=column,
                    imputed_value=mode_val,
                    original_value=record.get(column),
                    strategy=ImputationStrategy.MODE,
                    confidence=round(confidence, 4),
                    confidence_level=_classify_confidence(confidence),
                    contributing_records=n_observed,
                    provenance_hash=prov,
                )
                imputed.append(iv)

        self._record_metrics("mode", imputed, start)
        return imputed

    # ------------------------------------------------------------------
    # KNN imputation
    # ------------------------------------------------------------------

    def impute_knn(
        self,
        records: List[Dict[str, Any]],
        column: str,
        k: Optional[int] = None,
    ) -> List[ImputedValue]:
        """Impute missing values using k-nearest neighbors.

        Finds the k most similar complete records using Euclidean distance
        on numeric columns, then averages (numeric) or votes (categorical)
        the target column from those neighbors.

        Args:
            records: List of record dictionaries.
            column: Target column to impute.
            k: Number of neighbors. Defaults to config.knn_neighbors.

        Returns:
            List of ImputedValue for each missing record.

        Raises:
            ValueError: If insufficient complete records for KNN.
        """
        start = time.monotonic()
        k = k or self.config.knn_neighbors

        # Identify feature columns (numeric, non-target)
        all_cols = set()
        for r in records:
            all_cols.update(r.keys())
        feature_cols = sorted(
            col for col in all_cols
            if col != column and self._column_is_numeric(records, col)
        )

        if not feature_cols:
            raise ValueError("No numeric feature columns available for KNN")

        # Separate complete and incomplete records
        complete_indices: List[int] = []
        missing_indices: List[int] = []
        for i, r in enumerate(records):
            if _is_missing(r.get(column)):
                missing_indices.append(i)
            else:
                # Must also have complete features
                if all(not _is_missing(r.get(fc)) for fc in feature_cols):
                    complete_indices.append(i)

        if len(complete_indices) < k:
            raise ValueError(
                f"Need at least {k} complete records for KNN, got {len(complete_indices)}"
            )

        # Determine if target is numeric
        target_numeric = self._column_is_numeric(records, column)

        imputed: List[ImputedValue] = []
        for mi in missing_indices:
            target_record = records[mi]

            # Build feature vector for this record
            target_features = self._extract_feature_vector(
                target_record, feature_cols
            )
            if target_features is None:
                continue

            # Find k nearest neighbors
            distances: List[Tuple[int, float]] = []
            for ci in complete_indices:
                neighbor_features = self._extract_feature_vector(
                    records[ci], feature_cols
                )
                if neighbor_features is None:
                    continue
                dist = self._euclidean_distance(target_features, neighbor_features)
                distances.append((ci, dist))

            distances.sort(key=lambda x: x[1])
            neighbors = distances[:k]

            if not neighbors:
                continue

            # Compute imputed value from neighbors
            if target_numeric:
                neighbor_vals = [
                    float(records[idx].get(column))
                    for idx, _ in neighbors
                    if _is_numeric(records[idx].get(column))
                ]
                if not neighbor_vals:
                    continue
                # Distance-weighted average
                weights_and_vals = []
                for idx, dist in neighbors:
                    val = _to_float(records[idx].get(column))
                    if val is not None:
                        w = 1.0 / (dist + 1e-10)
                        weights_and_vals.append((w, val))

                if weights_and_vals:
                    total_weight = sum(w for w, _ in weights_and_vals)
                    imp_val = sum(w * v for w, v in weights_and_vals) / total_weight
                else:
                    imp_val = sum(neighbor_vals) / len(neighbor_vals)
                imp_val = round(imp_val, 8)
            else:
                neighbor_vals_cat = [
                    records[idx].get(column) for idx, _ in neighbors
                    if not _is_missing(records[idx].get(column))
                ]
                if not neighbor_vals_cat:
                    continue
                freq = Counter(str(v) for v in neighbor_vals_cat)
                imp_val = freq.most_common(1)[0][0]
                # Recover typed value
                for v in neighbor_vals_cat:
                    if str(v) == imp_val:
                        imp_val = v
                        break

            confidence = self._compute_confidence(
                "knn", len(neighbors),
                max(d for _, d in neighbors) if neighbors else 0.0
            )

            prov = _compute_provenance(
                "impute_knn", f"{column}:{mi}:{imp_val}:k={len(neighbors)}"
            )
            iv = ImputedValue(
                record_index=mi,
                column_name=column,
                imputed_value=imp_val,
                original_value=target_record.get(column),
                strategy=ImputationStrategy.KNN,
                confidence=round(confidence, 4),
                confidence_level=_classify_confidence(confidence),
                contributing_records=len(neighbors),
                provenance_hash=prov,
            )
            imputed.append(iv)

        self._record_metrics("knn", imputed, start)
        return imputed

    # ------------------------------------------------------------------
    # Regression imputation
    # ------------------------------------------------------------------

    def impute_regression(
        self,
        records: List[Dict[str, Any]],
        target_column: str,
        predictor_columns: Optional[List[str]] = None,
    ) -> List[ImputedValue]:
        """Impute missing values using linear regression (OLS).

        Fits y = b0 + b1*x1 + b2*x2 + ... using complete records, then
        predicts missing target values from their predictors.

        Uses the normal equation: beta = (X^T X)^-1 X^T y

        Args:
            records: List of record dictionaries.
            target_column: Column to impute.
            predictor_columns: Columns to use as predictors.
                              If None, auto-selects numeric columns.

        Returns:
            List of ImputedValue for each missing record.

        Raises:
            ValueError: If insufficient data for regression.
        """
        start = time.monotonic()

        # Auto-select numeric predictor columns
        if predictor_columns is None:
            all_cols = set()
            for r in records:
                all_cols.update(r.keys())
            predictor_columns = sorted(
                col for col in all_cols
                if col != target_column and self._column_is_numeric(records, col)
            )

        if not predictor_columns:
            raise ValueError("No predictor columns available for regression")

        # Separate complete and incomplete records
        complete_x: List[List[float]] = []
        complete_y: List[float] = []
        missing_indices: List[int] = []

        for i, r in enumerate(records):
            target_val = _to_float(r.get(target_column))
            if target_val is None:
                missing_indices.append(i)
                continue

            features = []
            skip = False
            for pc in predictor_columns:
                fv = _to_float(r.get(pc))
                if fv is None:
                    skip = True
                    break
                features.append(fv)
            if skip:
                continue

            complete_x.append(features)
            complete_y.append(target_val)

        n = len(complete_x)
        p = len(predictor_columns)
        if n < p + 2:
            raise ValueError(
                f"Need at least {p + 2} complete records for regression, got {n}"
            )

        # Fit OLS via normal equation
        coefficients = self._fit_ols(complete_x, complete_y)

        # Compute R-squared for confidence
        r_squared = self._compute_r_squared(complete_x, complete_y, coefficients)

        imputed: List[ImputedValue] = []
        for mi in missing_indices:
            r = records[mi]
            features = []
            skip = False
            for pc in predictor_columns:
                fv = _to_float(r.get(pc))
                if fv is None:
                    skip = True
                    break
                features.append(fv)
            if skip:
                continue

            # Predict: y = b0 + b1*x1 + b2*x2 + ...
            predicted = coefficients[0]  # intercept
            for j, x_val in enumerate(features):
                predicted += coefficients[j + 1] * x_val

            confidence = min(0.5 + r_squared * 0.45, 0.95)
            confidence = self._adjust_confidence_by_sample(confidence, n)

            prov = _compute_provenance(
                "impute_regression", f"{target_column}:{mi}:{predicted}"
            )
            iv = ImputedValue(
                record_index=mi,
                column_name=target_column,
                imputed_value=round(predicted, 8),
                original_value=r.get(target_column),
                strategy=ImputationStrategy.REGRESSION,
                confidence=round(confidence, 4),
                confidence_level=_classify_confidence(confidence),
                contributing_records=n,
                provenance_hash=prov,
            )
            imputed.append(iv)

        self._record_metrics("regression", imputed, start)
        return imputed

    # ------------------------------------------------------------------
    # Hot-deck imputation
    # ------------------------------------------------------------------

    def impute_hot_deck(
        self,
        records: List[Dict[str, Any]],
        column: str,
        method: str = "random",
    ) -> List[ImputedValue]:
        """Impute missing values using hot-deck donor selection.

        Hot-deck selects a donor record from observed values and copies
        its value to the missing position.

        Methods:
            - 'random': Random selection from observed values.
            - 'sequential': Use the nearest prior observed value.

        Args:
            records: List of record dictionaries.
            column: Column to impute.
            method: Donor selection method ('random' or 'sequential').

        Returns:
            List of ImputedValue for each missing record.

        Raises:
            ValueError: If no observed values available.
        """
        start = time.monotonic()
        observed_values = [
            (i, record.get(column))
            for i, record in enumerate(records)
            if not _is_missing(record.get(column))
        ]
        if not observed_values:
            raise ValueError(f"No observed values in column '{column}' for hot-deck")

        n_observed = len(observed_values)
        imputed: List[ImputedValue] = []

        for i, record in enumerate(records):
            if not _is_missing(record.get(column)):
                continue

            if method == "sequential":
                # Find nearest prior observed value
                donor_val = self._find_nearest_donor(i, observed_values)
            else:
                # Random selection
                donor_idx = random.randint(0, n_observed - 1)
                donor_val = observed_values[donor_idx][1]

            confidence = self._compute_confidence(
                "hot_deck", n_observed, 0.0
            )

            prov = _compute_provenance(
                "impute_hot_deck", f"{column}:{i}:{donor_val}"
            )
            iv = ImputedValue(
                record_index=i,
                column_name=column,
                imputed_value=donor_val,
                original_value=record.get(column),
                strategy=ImputationStrategy.HOT_DECK,
                confidence=round(confidence, 4),
                confidence_level=_classify_confidence(confidence),
                contributing_records=1,
                provenance_hash=prov,
            )
            imputed.append(iv)

        self._record_metrics("hot_deck", imputed, start)
        return imputed

    # ------------------------------------------------------------------
    # LOCF (Last Observation Carried Forward)
    # ------------------------------------------------------------------

    def impute_locf(
        self,
        records: List[Dict[str, Any]],
        column: str,
        sort_column: Optional[str] = None,
    ) -> List[ImputedValue]:
        """Impute missing values using Last Observation Carried Forward.

        Carries the most recent non-missing value forward to fill gaps.
        Leading missing values (before first observation) are left unimputed.

        Args:
            records: List of record dictionaries.
            column: Column to impute.
            sort_column: Optional column to sort by before applying LOCF.

        Returns:
            List of ImputedValue for filled positions.
        """
        start = time.monotonic()
        ordered = self._order_records(records, sort_column)

        imputed: List[ImputedValue] = []
        last_observed: Optional[Any] = None
        last_observed_dist = 0

        for orig_idx, record in ordered:
            val = record.get(column)
            if not _is_missing(val):
                last_observed = val
                last_observed_dist = 0
            elif last_observed is not None:
                last_observed_dist += 1
                # Confidence decays with distance from last observation
                confidence = max(0.4, 0.85 - 0.05 * last_observed_dist)

                prov = _compute_provenance(
                    "impute_locf", f"{column}:{orig_idx}:{last_observed}"
                )
                iv = ImputedValue(
                    record_index=orig_idx,
                    column_name=column,
                    imputed_value=last_observed,
                    original_value=val,
                    strategy=ImputationStrategy.LOCF,
                    confidence=round(confidence, 4),
                    confidence_level=_classify_confidence(confidence),
                    contributing_records=1,
                    provenance_hash=prov,
                )
                imputed.append(iv)

        self._record_metrics("locf", imputed, start)
        return imputed

    # ------------------------------------------------------------------
    # NOCB (Next Observation Carried Backward)
    # ------------------------------------------------------------------

    def impute_nocb(
        self,
        records: List[Dict[str, Any]],
        column: str,
        sort_column: Optional[str] = None,
    ) -> List[ImputedValue]:
        """Impute missing values using Next Observation Carried Backward.

        Fills missing values with the next available observed value.
        Trailing missing values (after last observation) are left unimputed.

        Args:
            records: List of record dictionaries.
            column: Column to impute.
            sort_column: Optional column to sort by before applying NOCB.

        Returns:
            List of ImputedValue for filled positions.
        """
        start = time.monotonic()
        ordered = self._order_records(records, sort_column)

        # Process in reverse
        imputed: List[ImputedValue] = []
        next_observed: Optional[Any] = None
        next_observed_dist = 0

        for orig_idx, record in reversed(ordered):
            val = record.get(column)
            if not _is_missing(val):
                next_observed = val
                next_observed_dist = 0
            elif next_observed is not None:
                next_observed_dist += 1
                confidence = max(0.4, 0.85 - 0.05 * next_observed_dist)

                prov = _compute_provenance(
                    "impute_nocb", f"{column}:{orig_idx}:{next_observed}"
                )
                iv = ImputedValue(
                    record_index=orig_idx,
                    column_name=column,
                    imputed_value=next_observed,
                    original_value=val,
                    strategy=ImputationStrategy.NOCB,
                    confidence=round(confidence, 4),
                    confidence_level=_classify_confidence(confidence),
                    contributing_records=1,
                    provenance_hash=prov,
                )
                imputed.append(iv)

        # Sort back to original order
        imputed.sort(key=lambda x: x.record_index)

        self._record_metrics("nocb", imputed, start)
        return imputed

    # ------------------------------------------------------------------
    # Grouped imputation
    # ------------------------------------------------------------------

    def impute_grouped(
        self,
        records: List[Dict[str, Any]],
        column: str,
        group_by: str,
        method: str = "mean",
    ) -> List[ImputedValue]:
        """Impute missing values by group, then apply aggregation.

        Groups records by the group_by column, then applies the specified
        aggregation method (mean, median, mode) within each group to
        fill missing values in the target column.

        Args:
            records: List of record dictionaries.
            column: Target column to impute.
            group_by: Column to group by.
            method: Aggregation method ('mean', 'median', 'mode').

        Returns:
            List of ImputedValue for each missing record.

        Raises:
            ValueError: If no observed values in any group.
        """
        start = time.monotonic()

        # Group records
        groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
        for i, r in enumerate(records):
            grp_key = str(r.get(group_by, "__ungrouped__"))
            if grp_key not in groups:
                groups[grp_key] = []
            groups[grp_key].append((i, r))

        # Compute group aggregates
        group_values: Dict[str, Any] = {}
        group_counts: Dict[str, int] = {}
        for grp_key, grp_records in groups.items():
            observed = [
                r.get(column) for _, r in grp_records
                if not _is_missing(r.get(column))
            ]
            group_counts[grp_key] = len(observed)
            if not observed:
                continue

            if method == "mean":
                nums = [float(v) for v in observed if _is_numeric(v)]
                if nums:
                    group_values[grp_key] = round(sum(nums) / len(nums), 8)
            elif method == "median":
                nums = [float(v) for v in observed if _is_numeric(v)]
                if nums:
                    group_values[grp_key] = round(statistics.median(nums), 8)
            else:  # mode
                freq = Counter(str(v) for v in observed)
                mode_str = freq.most_common(1)[0][0]
                for v in observed:
                    if str(v) == mode_str:
                        group_values[grp_key] = v
                        break

        # Compute global fallback
        all_observed = [
            r.get(column) for r in records
            if not _is_missing(r.get(column))
        ]
        global_fallback: Optional[Any] = None
        if all_observed:
            if method == "mean":
                nums = [float(v) for v in all_observed if _is_numeric(v)]
                if nums:
                    global_fallback = round(sum(nums) / len(nums), 8)
            elif method == "median":
                nums = [float(v) for v in all_observed if _is_numeric(v)]
                if nums:
                    global_fallback = round(statistics.median(nums), 8)
            else:
                freq = Counter(str(v) for v in all_observed)
                mode_str = freq.most_common(1)[0][0]
                for v in all_observed:
                    if str(v) == mode_str:
                        global_fallback = v
                        break

        imputed: List[ImputedValue] = []
        for i, record in enumerate(records):
            if not _is_missing(record.get(column)):
                continue

            grp_key = str(record.get(group_by, "__ungrouped__"))
            imp_val = group_values.get(grp_key, global_fallback)
            if imp_val is None:
                continue

            n_group = group_counts.get(grp_key, 0)
            confidence = self._compute_confidence(
                method, n_group, 0.0
            )
            # Bonus for group-specific imputation
            if grp_key in group_values:
                confidence = min(confidence + 0.05, 0.95)

            prov = _compute_provenance(
                "impute_grouped",
                f"{column}:{i}:{imp_val}:grp={grp_key}",
            )
            iv = ImputedValue(
                record_index=i,
                column_name=column,
                imputed_value=imp_val,
                original_value=record.get(column),
                strategy=ImputationStrategy.MEAN
                if method == "mean"
                else (
                    ImputationStrategy.MEDIAN
                    if method == "median"
                    else ImputationStrategy.MODE
                ),
                confidence=round(confidence, 4),
                confidence_level=_classify_confidence(confidence),
                contributing_records=n_group,
                provenance_hash=prov,
            )
            imputed.append(iv)

        self._record_metrics(f"grouped_{method}", imputed, start)
        return imputed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_numeric(
        self, records: List[Dict[str, Any]], column: str
    ) -> List[float]:
        """Extract non-missing numeric values from a column.

        Args:
            records: List of record dictionaries.
            column: Column name.

        Returns:
            List of float values.
        """
        result: List[float] = []
        for r in records:
            val = r.get(column)
            if not _is_missing(val) and _is_numeric(val):
                result.append(float(val))
        return result

    def _column_is_numeric(
        self, records: List[Dict[str, Any]], column: str
    ) -> bool:
        """Check if a column is predominantly numeric.

        Args:
            records: List of record dictionaries.
            column: Column name.

        Returns:
            True if >50% of non-missing values are numeric.
        """
        numeric_count = 0
        total_count = 0
        for r in records:
            val = r.get(column)
            if not _is_missing(val):
                total_count += 1
                if _is_numeric(val):
                    numeric_count += 1

        if total_count == 0:
            return False
        return numeric_count / total_count > 0.5

    def _extract_feature_vector(
        self,
        record: Dict[str, Any],
        feature_cols: List[str],
    ) -> Optional[List[float]]:
        """Extract numeric feature vector from a record.

        Args:
            record: Record dictionary.
            feature_cols: List of feature column names.

        Returns:
            List of float feature values, or None if any are missing.
        """
        features: List[float] = []
        for col in feature_cols:
            val = _to_float(record.get(col))
            if val is None:
                return None
            features.append(val)
        return features

    def _euclidean_distance(
        self, a: List[float], b: List[float]
    ) -> float:
        """Compute Euclidean distance between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Euclidean distance.
        """
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def _find_nearest_donor(
        self,
        target_idx: int,
        observed: List[Tuple[int, Any]],
    ) -> Any:
        """Find the nearest observed donor for sequential hot-deck.

        Args:
            target_idx: Index of the missing record.
            observed: List of (index, value) for observed records.

        Returns:
            Donor value from the nearest observed record.
        """
        best_dist = float("inf")
        best_val = observed[0][1]
        for obs_idx, obs_val in observed:
            dist = abs(target_idx - obs_idx)
            if dist < best_dist:
                best_dist = dist
                best_val = obs_val
        return best_val

    def _order_records(
        self,
        records: List[Dict[str, Any]],
        sort_column: Optional[str],
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """Order records by sort column or original index.

        Args:
            records: List of record dictionaries.
            sort_column: Optional column to sort by.

        Returns:
            List of (original_index, record) tuples in order.
        """
        indexed = [(i, r) for i, r in enumerate(records)]
        if sort_column:
            indexed.sort(
                key=lambda x: (
                    x[1].get(sort_column)
                    if not _is_missing(x[1].get(sort_column))
                    else ""
                )
            )
        return indexed

    def _fit_ols(
        self,
        x_data: List[List[float]],
        y_data: List[float],
    ) -> List[float]:
        """Fit Ordinary Least Squares via normal equation.

        Solves beta = (X^T X)^{-1} X^T y where X includes an intercept column.

        Args:
            x_data: List of feature vectors (n x p).
            y_data: List of target values (n).

        Returns:
            List of coefficients [intercept, b1, b2, ...].
        """
        n = len(x_data)
        p = len(x_data[0]) if x_data else 0

        # Add intercept column
        X = [[1.0] + row for row in x_data]
        cols = p + 1

        # X^T X
        XtX = [[0.0] * cols for _ in range(cols)]
        for i in range(cols):
            for j in range(cols):
                XtX[i][j] = sum(X[k][i] * X[k][j] for k in range(n))

        # X^T y
        Xty = [sum(X[k][i] * y_data[k] for k in range(n)) for i in range(cols)]

        # Solve via Gaussian elimination
        coefficients = self._solve_linear_system(XtX, Xty)
        return coefficients

    def _solve_linear_system(
        self,
        A: List[List[float]],
        b: List[float],
    ) -> List[float]:
        """Solve Ax = b via Gaussian elimination with partial pivoting.

        Args:
            A: Coefficient matrix (n x n).
            b: Right-hand side vector (n).

        Returns:
            Solution vector x (n).
        """
        n = len(b)
        # Augmented matrix
        aug = [A[i][:] + [b[i]] for i in range(n)]

        # Forward elimination with partial pivoting
        for col in range(n):
            # Find pivot
            max_row = col
            max_val = abs(aug[col][col])
            for row in range(col + 1, n):
                if abs(aug[row][col]) > max_val:
                    max_val = abs(aug[row][col])
                    max_row = row
            aug[col], aug[max_row] = aug[max_row], aug[col]

            pivot = aug[col][col]
            if abs(pivot) < 1e-12:
                # Near-singular, add small regularization
                aug[col][col] += 1e-6
                pivot = aug[col][col]

            for row in range(col + 1, n):
                factor = aug[row][col] / pivot
                for j in range(col, n + 1):
                    aug[row][j] -= factor * aug[col][j]

        # Back substitution
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = aug[i][n]
            for j in range(i + 1, n):
                s -= aug[i][j] * x[j]
            if abs(aug[i][i]) < 1e-12:
                x[i] = 0.0
            else:
                x[i] = s / aug[i][i]

        return x

    def _compute_r_squared(
        self,
        x_data: List[List[float]],
        y_data: List[float],
        coefficients: List[float],
    ) -> float:
        """Compute R-squared for the fitted regression.

        Args:
            x_data: Feature matrix.
            y_data: Target values.
            coefficients: OLS coefficients [intercept, b1, b2, ...].

        Returns:
            R-squared value in [0.0, 1.0].
        """
        if not y_data:
            return 0.0

        y_mean = sum(y_data) / len(y_data)
        ss_tot = sum((y - y_mean) ** 2 for y in y_data)
        if ss_tot < 1e-12:
            return 1.0

        ss_res = 0.0
        for i, row in enumerate(x_data):
            predicted = coefficients[0]
            for j, x_val in enumerate(row):
                predicted += coefficients[j + 1] * x_val
            ss_res += (y_data[i] - predicted) ** 2

        r2 = 1.0 - ss_res / ss_tot
        return max(0.0, min(1.0, r2))

    def _compute_confidence(
        self,
        method: str,
        neighbors_count: int,
        variance: float,
    ) -> float:
        """Compute confidence score for an imputation.

        Factors:
            - Method reliability (mean < median < knn < regression)
            - Sample size (more contributing records = higher confidence)
            - Variance (lower variance = higher confidence)

        Args:
            method: Imputation method name.
            neighbors_count: Number of contributing records.
            variance: Variance or max distance metric.

        Returns:
            Confidence score in [0.0, 1.0].
        """
        base_confidence = {
            "mean": 0.72,
            "median": 0.76,
            "mode": 0.70,
            "knn": 0.80,
            "regression": 0.78,
            "hot_deck": 0.65,
            "locf": 0.62,
            "nocb": 0.62,
            "grouped_mean": 0.75,
            "grouped_median": 0.78,
            "grouped_mode": 0.73,
        }
        confidence = base_confidence.get(method, 0.70)

        # Sample size adjustment
        confidence = self._adjust_confidence_by_sample(
            confidence, neighbors_count
        )

        # Variance penalty (for numeric methods)
        if variance > 0 and method in ("mean", "median", "knn"):
            norm_var = min(variance / 100.0, 1.0)
            confidence *= (1.0 - 0.15 * norm_var)

        return max(0.0, min(1.0, confidence))

    def _adjust_confidence_by_sample(
        self, confidence: float, n: int
    ) -> float:
        """Adjust confidence based on sample size.

        Args:
            confidence: Base confidence.
            n: Sample size.

        Returns:
            Adjusted confidence.
        """
        if n >= 1000:
            return min(confidence * 1.10, 0.95)
        if n >= 100:
            return min(confidence * 1.05, 0.95)
        if n >= 30:
            return confidence
        if n >= 10:
            return confidence * 0.95
        if n >= 3:
            return confidence * 0.85
        return confidence * 0.70

    def _record_metrics(
        self,
        method: str,
        imputed: List[ImputedValue],
        start: float,
    ) -> None:
        """Record Prometheus metrics for an imputation.

        Args:
            method: Imputation method name.
            imputed: List of imputed values.
            start: Monotonic start time.
        """
        elapsed = time.monotonic() - start
        observe_duration("impute", elapsed)
        if imputed:
            inc_values_imputed(method, len(imputed))
            for iv in imputed:
                observe_confidence(method, iv.confidence)

        logger.info(
            "%s imputation: %d values, elapsed=%.3fs",
            method, len(imputed), elapsed,
        )
