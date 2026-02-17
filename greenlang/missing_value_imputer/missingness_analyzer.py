# -*- coding: utf-8 -*-
"""
Missingness Analyzer Engine - AGENT-DATA-012: Missing Value Imputer (GL-DATA-X-015)

Analyzes missing data patterns across datasets. Detects missingness per column,
classifies the mechanism (MCAR/MAR/MNAR) using Little's MCAR test approximation
and correlation analysis, computes binary missing-pattern matrices, and recommends
imputation strategies based on data type, pattern, and missingness mechanism.

Zero-Hallucination Guarantees:
    - All statistics are deterministic Python arithmetic
    - MCAR/MAR/MNAR classification uses correlation and variance heuristics only
    - No ML/LLM calls in the analysis path
    - SHA-256 provenance on every analysis output
    - Thread-safe operation

Example:
    >>> from greenlang.missing_value_imputer.missingness_analyzer import MissingnessAnalyzerEngine
    >>> from greenlang.missing_value_imputer.config import MissingValueImputerConfig
    >>> engine = MissingnessAnalyzerEngine(MissingValueImputerConfig())
    >>> records = [
    ...     {"a": 1, "b": None, "c": "x"},
    ...     {"a": None, "b": 2.0, "c": "y"},
    ... ]
    >>> report = engine.analyze_dataset(records)
    >>> print(report.columns_with_missing, report.complete_record_pct)

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
from typing import Any, Dict, List, Optional, Set, Tuple

from greenlang.missing_value_imputer.config import MissingValueImputerConfig
from greenlang.missing_value_imputer.models import (
    ColumnAnalysis,
    ConfidenceLevel,
    DataColumnType,
    ImputationStrategy,
    MissingnessPattern,
    MissingnessReport,
    MissingnessType,
    PatternType,
    StrategySelection,
)
from greenlang.missing_value_imputer.metrics import (
    inc_analyses,
    observe_duration,
    inc_errors,
)
from greenlang.missing_value_imputer.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

__all__ = [
    "MissingnessAnalyzerEngine",
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
        True if the value is None, empty string, whitespace-only, or NaN.
    """
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _is_numeric(value: Any) -> bool:
    """Check if a value is numeric (int or float, not bool).

    Args:
        value: The value to check.

    Returns:
        True if value is int or float (excluding bool).
    """
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


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
        return statistics.stdev([float(v) for v in values])
    except (ValueError, TypeError, AttributeError, statistics.StatisticsError):
        return 0.0


def _safe_mean(values: List[float]) -> float:
    """Compute mean, returning 0.0 for empty list.

    Args:
        values: List of numeric values.

    Returns:
        Arithmetic mean or 0.0.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_median(values: List[float]) -> float:
    """Compute median, returning 0.0 for empty list.

    Args:
        values: List of numeric values.

    Returns:
        Median or 0.0.
    """
    if not values:
        return 0.0
    return statistics.median(values)


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


def _detect_column_type(values: List[Any]) -> DataColumnType:
    """Detect the data type of a column from its non-missing values.

    Args:
        values: List of non-missing values.

    Returns:
        Detected DataColumnType enum value.
    """
    if not values:
        return DataColumnType.TEXT

    type_counts: Dict[str, int] = {
        "numeric": 0,
        "boolean": 0,
        "datetime": 0,
        "text": 0,
    }

    for v in values:
        if isinstance(v, bool):
            type_counts["boolean"] += 1
        elif isinstance(v, (int, float)):
            type_counts["numeric"] += 1
        elif isinstance(v, datetime):
            type_counts["datetime"] += 1
        else:
            type_counts["text"] += 1

    max_type = max(type_counts, key=lambda k: type_counts[k])
    type_map = {
        "numeric": DataColumnType.NUMERIC,
        "boolean": DataColumnType.BOOLEAN,
        "datetime": DataColumnType.DATETIME,
        "text": DataColumnType.TEXT,
    }
    return type_map.get(max_type, DataColumnType.TEXT)


def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient between two series.

    Returns 0.0 if insufficient data or zero variance.

    Args:
        x: First numeric series.
        y: Second numeric series.

    Returns:
        Pearson r in [-1.0, 1.0] or 0.0 on error.
    """
    n = min(len(x), len(y))
    if n < 3:
        return 0.0

    x = x[:n]
    y = y[:n]
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    denom = math.sqrt(var_x * var_y)
    if denom < 1e-12:
        return 0.0

    return cov / denom


# ===========================================================================
# MissingnessAnalyzerEngine
# ===========================================================================


class MissingnessAnalyzerEngine:
    """Analyzes missing data patterns and recommends imputation strategies.

    This engine examines a dataset represented as a list of dictionaries and
    produces a comprehensive missingness analysis covering per-column stats,
    pattern classification, missingness mechanism detection, and strategy
    recommendations.

    Zero-hallucination: All computations use deterministic Python arithmetic.
    No LLM or ML calls exist in any analysis code path.

    Attributes:
        config: Service configuration.
        provenance: SHA-256 provenance tracker.

    Example:
        >>> engine = MissingnessAnalyzerEngine(MissingValueImputerConfig())
        >>> report = engine.analyze_dataset([{"a": 1, "b": None}])
        >>> assert report.columns_with_missing >= 0
    """

    def __init__(self, config: MissingValueImputerConfig) -> None:
        """Initialize the MissingnessAnalyzerEngine.

        Args:
            config: Service configuration instance.
        """
        self.config = config
        self.provenance = ProvenanceTracker()
        logger.info("MissingnessAnalyzerEngine initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_dataset(
        self,
        records: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
    ) -> MissingnessReport:
        """Analyze missingness across a dataset.

        Computes per-column missing counts, overall statistics, pattern
        classification, and missingness mechanism for every column.

        Args:
            records: List of record dictionaries.
            columns: Optional subset of columns to analyze.
                     When None, all columns from all records are analyzed.

        Returns:
            MissingnessReport with complete analysis.

        Raises:
            ValueError: If records list is empty.
        """
        start = time.monotonic()

        if not records:
            raise ValueError("records must be non-empty for analysis")

        all_columns = self._collect_columns(records)
        target_columns = columns if columns else sorted(all_columns)

        column_analyses: List[ColumnAnalysis] = []
        for col in target_columns:
            analysis = self.get_column_analysis(records, col)
            column_analyses.append(analysis)

        # Compute pattern and correlations
        pattern_type = self.detect_pattern_type(records)
        correlations = self.compute_missing_correlations(records)

        # Determine overall missingness type
        missingness_types = [ca.missingness_type for ca in column_analyses]
        overall_missingness = self._aggregate_missingness_type(missingness_types)

        # Overall stats
        total_records = len(records)
        total_columns = len(target_columns)
        total_cells = total_records * total_columns
        total_missing = sum(ca.missing_count for ca in column_analyses)
        columns_with_missing = sum(1 for ca in column_analyses if ca.missing_count > 0)
        complete_records = self._count_complete_records(records, target_columns)
        complete_pct = complete_records / total_records if total_records > 0 else 0.0
        overall_missing_pct = total_missing / total_cells if total_cells > 0 else 0.0

        # Build pattern model
        affected_columns = [
            ca.column_name for ca in column_analyses if ca.missing_count > 0
        ]
        pattern = MissingnessPattern(
            pattern_type=pattern_type,
            missingness_type=overall_missingness,
            affected_columns=affected_columns,
            correlation_matrix=correlations,
            total_missing=total_missing,
            total_cells=total_cells,
            overall_missing_pct=round(overall_missing_pct, 6),
            provenance_hash=_compute_provenance(
                "pattern_analysis", str(total_missing)
            ),
        )

        # Build report
        provenance_hash = _compute_provenance(
            "analyze_dataset", f"{total_records}:{total_columns}:{total_missing}"
        )

        report = MissingnessReport(
            pattern=pattern,
            columns=column_analyses,
            total_records=total_records,
            total_columns=total_columns,
            columns_with_missing=columns_with_missing,
            complete_records=complete_records,
            complete_record_pct=round(complete_pct, 6),
            provenance_hash=provenance_hash,
        )

        # Record provenance
        self.provenance.record(
            "analysis", report.report_id, "analyze_dataset", provenance_hash
        )

        elapsed = time.monotonic() - start
        observe_duration("analyze", elapsed)
        inc_analyses(overall_missingness.value)

        logger.info(
            "Dataset analyzed: records=%d columns=%d missing=%d (%.1f%%) "
            "pattern=%s mechanism=%s elapsed=%.3fs",
            total_records,
            total_columns,
            total_missing,
            overall_missing_pct * 100,
            pattern_type.value,
            overall_missingness.value,
            elapsed,
        )
        return report

    def classify_missingness(
        self,
        column_data: List[Any],
        full_data: List[Dict[str, Any]],
    ) -> MissingnessType:
        """Classify the missingness mechanism for a single column.

        Uses Little's MCAR test approximation: compares the distribution of
        other columns between records where this column is missing vs observed.
        Falls back to correlation-based heuristics when insufficient data.

        Heuristics:
            - MCAR: Missingness is uncorrelated with observed values.
              The variance of missingness rates across groups is low (< 0.1).
            - MAR: Missingness correlates with other observed columns.
              Average absolute correlation > 0.3 with at least one other column.
            - MNAR: Missingness correlates with the value itself.
              Missing values cluster at extremes (top/bottom quartiles).

        Args:
            column_data: Values for the target column (may include None).
            full_data: Full dataset as list of record dicts.

        Returns:
            MissingnessType classification.
        """
        if not column_data or not full_data:
            return MissingnessType.UNKNOWN

        missing_indices: Set[int] = set()
        observed_indices: Set[int] = set()
        for i, v in enumerate(column_data):
            if _is_missing(v):
                missing_indices.add(i)
            else:
                observed_indices.add(i)

        if not missing_indices or not observed_indices:
            return MissingnessType.UNKNOWN

        missing_count = len(missing_indices)
        total = len(column_data)

        # Check MNAR: is missingness related to the value itself?
        if self._check_mnar(column_data, missing_indices, observed_indices):
            return MissingnessType.MNAR

        # Check MAR: is missingness related to other columns?
        if self._check_mar(full_data, missing_indices, observed_indices):
            return MissingnessType.MAR

        # Approximate Little's MCAR test: uniform missing rate across groups
        if self._check_mcar(full_data, missing_indices, total):
            return MissingnessType.MCAR

        return MissingnessType.UNKNOWN

    def compute_pattern_matrix(
        self, records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute binary pattern matrix of missingness.

        Each unique pattern is a tuple of 0/1 indicators across columns.
        The matrix reveals whether missingness follows a monotone or
        arbitrary structure.

        Args:
            records: List of record dictionaries.

        Returns:
            Dictionary with keys:
                - patterns: List of pattern dicts (pattern tuple -> count)
                - columns: List of column names in order
                - n_patterns: Number of unique missing patterns
                - is_monotone: Whether patterns are monotone
                - pattern_frequencies: Most common patterns
                - provenance_hash: SHA-256 hash
        """
        if not records:
            return {
                "patterns": [],
                "columns": [],
                "n_patterns": 0,
                "is_monotone": False,
                "pattern_frequencies": {},
                "provenance_hash": _compute_provenance("pattern_matrix", "empty"),
            }

        all_cols = sorted(self._collect_columns(records))
        pattern_counter: Counter = Counter()

        for record in records:
            pattern = tuple(
                0 if _is_missing(record.get(col)) else 1 for col in all_cols
            )
            pattern_counter[pattern] += 1

        # Check monotone: can columns be reordered so missing patterns nest
        is_monotone = self._check_monotone(records, all_cols)

        patterns = [
            {"pattern": list(p), "count": c}
            for p, c in pattern_counter.most_common()
        ]

        freq = {str(list(p)): c for p, c in pattern_counter.most_common(20)}

        provenance_hash = _compute_provenance(
            "compute_pattern_matrix", str(len(patterns))
        )

        return {
            "patterns": patterns,
            "columns": all_cols,
            "n_patterns": len(pattern_counter),
            "is_monotone": is_monotone,
            "pattern_frequencies": freq,
            "provenance_hash": provenance_hash,
        }

    def get_column_analysis(
        self,
        records: List[Dict[str, Any]],
        column: str,
    ) -> ColumnAnalysis:
        """Compute detailed missingness analysis for a single column.

        Args:
            records: List of record dictionaries.
            column: Name of the column to analyze.

        Returns:
            ColumnAnalysis model with complete statistics.

        Raises:
            ValueError: If records list is empty.
        """
        if not records:
            raise ValueError("records must be non-empty")

        values = [record.get(column) for record in records]
        total = len(values)
        missing_count = sum(1 for v in values if _is_missing(v))
        missing_pct = missing_count / total if total > 0 else 0.0

        non_missing = [v for v in values if not _is_missing(v)]
        unique_values = len(set(str(v) for v in non_missing))

        # Detect column type
        col_type = _detect_column_type(non_missing)

        # Compute stats
        mean_val: Optional[float] = None
        median_val: Optional[float] = None
        mode_val: Optional[Any] = None
        std_val: Optional[float] = None
        min_val: Optional[Any] = None
        max_val: Optional[Any] = None

        if non_missing:
            if col_type == DataColumnType.NUMERIC:
                numeric_vals = [
                    float(v) for v in non_missing if _is_numeric(v)
                ]
                if numeric_vals:
                    mean_val = round(_safe_mean(numeric_vals), 6)
                    median_val = round(_safe_median(numeric_vals), 6)
                    std_val = round(_safe_stdev(numeric_vals), 6)
                    min_val = min(numeric_vals)
                    max_val = max(numeric_vals)

            # Mode for all types
            freq = Counter(str(v) for v in non_missing)
            if freq:
                mode_val = freq.most_common(1)[0][0]

        # Classify missingness mechanism
        missingness_type = self.classify_missingness(values, records)

        # Recommend strategy
        recommended = self._recommend_column_strategy(
            col_type, missingness_type, missing_pct, total
        )

        provenance_hash = _compute_provenance(
            "column_analysis", f"{column}:{missing_count}:{total}"
        )

        return ColumnAnalysis(
            column_name=column,
            column_type=col_type,
            total_values=total,
            missing_count=missing_count,
            missing_pct=round(missing_pct, 6),
            missingness_type=missingness_type,
            unique_values=unique_values,
            mean_value=mean_val,
            median_value=median_val,
            mode_value=mode_val,
            std_dev=std_val,
            min_value=min_val,
            max_value=max_val,
            recommended_strategy=recommended,
            provenance_hash=provenance_hash,
        )

    def compute_missing_correlations(
        self, records: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute pairwise missingness correlation between columns.

        Builds a binary missingness indicator for each column (1 = missing,
        0 = present) and computes Pearson correlation between every pair.

        Args:
            records: List of record dictionaries.

        Returns:
            Nested dict: {col_a: {col_b: correlation_value}}.
        """
        if not records:
            return {}

        all_cols = sorted(self._collect_columns(records))
        if len(all_cols) < 2:
            return {}

        # Build binary missingness indicators
        indicators: Dict[str, List[float]] = {}
        for col in all_cols:
            indicators[col] = [
                1.0 if _is_missing(r.get(col)) else 0.0 for r in records
            ]

        # Compute pairwise correlations
        result: Dict[str, Dict[str, float]] = {}
        for col_a in all_cols:
            result[col_a] = {}
            for col_b in all_cols:
                if col_a == col_b:
                    result[col_a][col_b] = 1.0
                elif col_b in result and col_a in result[col_b]:
                    result[col_a][col_b] = result[col_b][col_a]
                else:
                    corr = _pearson_correlation(
                        indicators[col_a], indicators[col_b]
                    )
                    result[col_a][col_b] = round(corr, 6)

        return result

    def detect_pattern_type(
        self, records: List[Dict[str, Any]]
    ) -> PatternType:
        """Detect the overall missing data pattern type.

        Classification:
            - UNIVARIATE: Only one column has missing values.
            - MONOTONE: Columns can be ordered so missingness nests.
            - ARBITRARY: Multiple columns with no monotone ordering.
            - PLANNED: All missing in same records (skip pattern).

        Args:
            records: List of record dictionaries.

        Returns:
            PatternType enum value.
        """
        if not records:
            return PatternType.ARBITRARY

        all_cols = sorted(self._collect_columns(records))
        cols_with_missing: List[str] = []
        for col in all_cols:
            has_missing = any(_is_missing(r.get(col)) for r in records)
            if has_missing:
                cols_with_missing.append(col)

        if len(cols_with_missing) == 0:
            return PatternType.ARBITRARY

        if len(cols_with_missing) == 1:
            return PatternType.UNIVARIATE

        # Check for planned missingness (same rows always missing together)
        if self._check_planned(records, cols_with_missing):
            return PatternType.PLANNED

        # Check monotone
        if self._check_monotone(records, cols_with_missing):
            return PatternType.MONOTONE

        return PatternType.ARBITRARY

    def recommend_strategies(
        self, report: MissingnessReport
    ) -> Dict[str, StrategySelection]:
        """Recommend imputation strategy for each column with missing values.

        Uses column data type, missingness mechanism, missing percentage,
        and dataset size to auto-select the best strategy.

        Args:
            report: MissingnessReport from analyze_dataset.

        Returns:
            Dict mapping column name to StrategySelection.
        """
        result: Dict[str, StrategySelection] = {}

        for col_analysis in report.columns:
            if col_analysis.missing_count == 0:
                continue

            strategy = self._recommend_column_strategy(
                col_analysis.column_type,
                col_analysis.missingness_type,
                col_analysis.missing_pct,
                col_analysis.total_values,
            )

            alternatives = self._get_alternative_strategies(
                col_analysis.column_type,
                col_analysis.missingness_type,
                col_analysis.missing_pct,
            )

            rationale = self._build_rationale(
                col_analysis, strategy
            )

            estimated_confidence = self._estimate_confidence(
                strategy, col_analysis.missing_pct, col_analysis.missingness_type
            )

            provenance_hash = _compute_provenance(
                "recommend_strategy",
                f"{col_analysis.column_name}:{strategy.value}",
            )

            selection = StrategySelection(
                column_name=col_analysis.column_name,
                recommended_strategy=strategy,
                alternative_strategies=alternatives,
                rationale=rationale,
                estimated_confidence=round(estimated_confidence, 4),
                column_type=col_analysis.column_type,
                missing_pct=col_analysis.missing_pct,
                provenance_hash=provenance_hash,
            )
            result[col_analysis.column_name] = selection

        logger.info(
            "Strategy recommendations generated for %d columns", len(result)
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_columns(self, records: List[Dict[str, Any]]) -> Set[str]:
        """Collect all unique column names across records.

        Args:
            records: List of record dictionaries.

        Returns:
            Set of column names.
        """
        columns: Set[str] = set()
        for record in records:
            columns.update(record.keys())
        return columns

    def _count_complete_records(
        self,
        records: List[Dict[str, Any]],
        columns: List[str],
    ) -> int:
        """Count records with no missing values in the target columns.

        Args:
            records: List of record dictionaries.
            columns: List of columns to check.

        Returns:
            Number of complete records.
        """
        count = 0
        for record in records:
            is_complete = all(
                not _is_missing(record.get(col)) for col in columns
            )
            if is_complete:
                count += 1
        return count

    def _check_mnar(
        self,
        column_data: List[Any],
        missing_indices: Set[int],
        observed_indices: Set[int],
    ) -> bool:
        """Check for Missing Not At Random (MNAR) pattern.

        MNAR is suspected when missing values correlate with the column's
        own value distribution (e.g., extreme values are more likely missing).

        Args:
            column_data: Values for the target column.
            missing_indices: Indices where value is missing.
            observed_indices: Indices where value is observed.

        Returns:
            True if MNAR pattern is detected.
        """
        observed_vals = [
            column_data[i] for i in observed_indices
            if _is_numeric(column_data[i])
        ]
        if len(observed_vals) < 10:
            return False

        observed_vals_sorted = sorted(observed_vals)
        n = len(observed_vals_sorted)
        q1_threshold = observed_vals_sorted[n // 4]
        q3_threshold = observed_vals_sorted[(3 * n) // 4]

        # Check if adjacent-to-missing records cluster at extremes
        # Use records just before or after missing indices
        total = len(column_data)
        adjacent_count = 0
        extreme_count = 0
        for idx in missing_indices:
            for neighbor in [idx - 1, idx + 1]:
                if 0 <= neighbor < total and neighbor in observed_indices:
                    val = column_data[neighbor]
                    if _is_numeric(val):
                        adjacent_count += 1
                        if val <= q1_threshold or val >= q3_threshold:
                            extreme_count += 1

        if adjacent_count < 5:
            return False

        extreme_ratio = extreme_count / adjacent_count
        # If >70% of adjacent values are in extreme quartiles, suspect MNAR
        return extreme_ratio > 0.70

    def _check_mar(
        self,
        full_data: List[Dict[str, Any]],
        missing_indices: Set[int],
        observed_indices: Set[int],
    ) -> bool:
        """Check for Missing At Random (MAR) pattern.

        MAR is suspected when missingness in the target column correlates
        with values in other observed columns.

        Args:
            full_data: Full dataset.
            missing_indices: Indices where target is missing.
            observed_indices: Indices where target is observed.

        Returns:
            True if MAR pattern is detected.
        """
        if not full_data or len(full_data) < 10:
            return False

        all_cols = sorted(self._collect_columns(full_data))
        if len(all_cols) < 2:
            return False

        # Build missingness indicator for target
        indicator = [
            1.0 if i in missing_indices else 0.0
            for i in range(len(full_data))
        ]

        significant_correlations = 0
        tested_cols = 0

        for col in all_cols:
            col_values = [r.get(col) for r in full_data]
            numeric_vals = []
            for v in col_values:
                if _is_numeric(v):
                    numeric_vals.append(float(v))
                else:
                    numeric_vals.append(0.0)

            # Only test if column has sufficient variance
            if _safe_stdev(numeric_vals) < 1e-10:
                continue

            tested_cols += 1
            corr = abs(_pearson_correlation(indicator, numeric_vals))
            if corr > 0.3:
                significant_correlations += 1

        if tested_cols == 0:
            return False

        # MAR if at least one strong correlation found
        return significant_correlations >= 1

    def _check_mcar(
        self,
        full_data: List[Dict[str, Any]],
        missing_indices: Set[int],
        total: int,
    ) -> bool:
        """Check for Missing Completely At Random (MCAR) pattern.

        Uses an approximation of Little's MCAR test: checks if the
        missing rate is uniform across subgroups of other variables.

        Args:
            full_data: Full dataset.
            missing_indices: Indices where target column is missing.
            total: Total number of values.

        Returns:
            True if MCAR pattern is detected.
        """
        if total < 10:
            return True  # Default to MCAR for very small datasets

        all_cols = sorted(self._collect_columns(full_data))
        group_missing_rates: List[float] = []

        for col in all_cols:
            # Split data into two groups by median of this column
            numeric_vals: List[Tuple[int, float]] = []
            for i, r in enumerate(full_data):
                val = r.get(col)
                if _is_numeric(val):
                    numeric_vals.append((i, float(val)))

            if len(numeric_vals) < 4:
                continue

            numeric_vals.sort(key=lambda x: x[1])
            mid = len(numeric_vals) // 2
            group_low = {idx for idx, _ in numeric_vals[:mid]}
            group_high = {idx for idx, _ in numeric_vals[mid:]}

            rate_low = (
                sum(1 for idx in group_low if idx in missing_indices) /
                len(group_low)
            ) if group_low else 0.0
            rate_high = (
                sum(1 for idx in group_high if idx in missing_indices) /
                len(group_high)
            ) if group_high else 0.0

            group_missing_rates.append(rate_low)
            group_missing_rates.append(rate_high)

        if len(group_missing_rates) < 2:
            return True

        # MCAR if variance of group-level missing rates is low
        rate_stdev = _safe_stdev(group_missing_rates)
        return rate_stdev < 0.10

    def _check_monotone(
        self,
        records: List[Dict[str, Any]],
        columns: List[str],
    ) -> bool:
        """Check whether missing patterns are monotone.

        Monotone: there exists an ordering of columns such that for each
        record, once a column is missing all subsequent columns are also missing.

        Args:
            records: List of record dictionaries.
            columns: List of columns to check.

        Returns:
            True if monotone pattern detected.
        """
        if len(columns) < 2:
            return True

        # Sort columns by missing count (ascending)
        missing_counts = []
        for col in columns:
            mc = sum(1 for r in records if _is_missing(r.get(col)))
            missing_counts.append((col, mc))
        missing_counts.sort(key=lambda x: x[1])
        sorted_cols = [c for c, _ in missing_counts]

        # Check monotone property
        for record in records:
            found_missing = False
            for col in sorted_cols:
                if _is_missing(record.get(col)):
                    found_missing = True
                elif found_missing:
                    return False

        return True

    def _check_planned(
        self,
        records: List[Dict[str, Any]],
        columns_with_missing: List[str],
    ) -> bool:
        """Check whether missingness is planned (skip pattern).

        Planned: Missing values in different columns always appear in
        exactly the same set of records.

        Args:
            records: List of record dictionaries.
            columns_with_missing: Columns that have at least one missing value.

        Returns:
            True if planned pattern detected.
        """
        if len(columns_with_missing) < 2:
            return False

        # Get missing row sets for each column
        missing_sets: List[frozenset] = []
        for col in columns_with_missing:
            missing_rows = frozenset(
                i for i, r in enumerate(records) if _is_missing(r.get(col))
            )
            missing_sets.append(missing_rows)

        # Planned if all columns have identical missing row sets
        first_set = missing_sets[0]
        return all(s == first_set for s in missing_sets)

    def _aggregate_missingness_type(
        self, types: List[MissingnessType]
    ) -> MissingnessType:
        """Aggregate column-level missingness types into an overall type.

        Priority: MNAR > MAR > MCAR > UNKNOWN.

        Args:
            types: List of per-column MissingnessType values.

        Returns:
            Overall MissingnessType.
        """
        if not types:
            return MissingnessType.UNKNOWN

        if MissingnessType.MNAR in types:
            return MissingnessType.MNAR
        if MissingnessType.MAR in types:
            return MissingnessType.MAR
        if MissingnessType.MCAR in types:
            return MissingnessType.MCAR
        return MissingnessType.UNKNOWN

    def _recommend_column_strategy(
        self,
        col_type: DataColumnType,
        missingness_type: MissingnessType,
        missing_pct: float,
        total_records: int,
    ) -> ImputationStrategy:
        """Recommend best imputation strategy for a single column.

        Decision tree:
            1. High missing pct (>50%) -> MICE or regulatory_default
            2. MNAR -> regression or rule_based
            3. MAR -> knn or regression
            4. MCAR numeric -> median (robust to outliers)
            5. MCAR categorical -> mode
            6. Time-series columns -> linear_interpolation

        Args:
            col_type: Column data type.
            missingness_type: Detected missingness mechanism.
            missing_pct: Fraction of missing values.
            total_records: Total number of records.

        Returns:
            Recommended ImputationStrategy.
        """
        # Very high missingness -> MICE or regulatory defaults
        if missing_pct > 0.50:
            if self.config.enable_ml_imputation and total_records >= 100:
                return ImputationStrategy.MICE
            return ImputationStrategy.REGULATORY_DEFAULT

        # MNAR needs conditional methods
        if missingness_type == MissingnessType.MNAR:
            if col_type == DataColumnType.NUMERIC:
                return ImputationStrategy.REGRESSION
            return ImputationStrategy.RULE_BASED

        # MAR benefits from conditional imputation
        if missingness_type == MissingnessType.MAR:
            if col_type == DataColumnType.NUMERIC and total_records >= 50:
                return ImputationStrategy.KNN
            return ImputationStrategy.REGRESSION

        # MCAR or UNKNOWN - simple methods are appropriate
        if col_type == DataColumnType.NUMERIC:
            return ImputationStrategy.MEDIAN
        if col_type == DataColumnType.CATEGORICAL:
            return ImputationStrategy.MODE
        if col_type == DataColumnType.BOOLEAN:
            return ImputationStrategy.MODE
        if col_type == DataColumnType.DATETIME:
            if self.config.enable_timeseries:
                return ImputationStrategy.LINEAR_INTERPOLATION
            return ImputationStrategy.LOCF
        # TEXT
        return ImputationStrategy.MODE

    def _get_alternative_strategies(
        self,
        col_type: DataColumnType,
        missingness_type: MissingnessType,
        missing_pct: float,
    ) -> List[ImputationStrategy]:
        """Get ranked alternative strategies for a column.

        Args:
            col_type: Column data type.
            missingness_type: Missingness mechanism.
            missing_pct: Fraction of missing values.

        Returns:
            List of alternative ImputationStrategy values.
        """
        alternatives: List[ImputationStrategy] = []

        if col_type == DataColumnType.NUMERIC:
            alternatives = [
                ImputationStrategy.MEAN,
                ImputationStrategy.MEDIAN,
                ImputationStrategy.KNN,
                ImputationStrategy.REGRESSION,
                ImputationStrategy.RANDOM_FOREST,
                ImputationStrategy.MICE,
            ]
        elif col_type == DataColumnType.CATEGORICAL:
            alternatives = [
                ImputationStrategy.MODE,
                ImputationStrategy.KNN,
                ImputationStrategy.HOT_DECK,
                ImputationStrategy.RULE_BASED,
            ]
        elif col_type == DataColumnType.DATETIME:
            alternatives = [
                ImputationStrategy.LINEAR_INTERPOLATION,
                ImputationStrategy.SPLINE_INTERPOLATION,
                ImputationStrategy.LOCF,
                ImputationStrategy.NOCB,
            ]
        elif col_type == DataColumnType.BOOLEAN:
            alternatives = [
                ImputationStrategy.MODE,
                ImputationStrategy.RULE_BASED,
            ]
        else:
            alternatives = [
                ImputationStrategy.MODE,
                ImputationStrategy.HOT_DECK,
                ImputationStrategy.RULE_BASED,
            ]

        # Filter to top 4
        return alternatives[:4]

    def _build_rationale(
        self,
        col_analysis: ColumnAnalysis,
        strategy: ImputationStrategy,
    ) -> str:
        """Build human-readable rationale for a strategy recommendation.

        Args:
            col_analysis: Column analysis data.
            strategy: Recommended strategy.

        Returns:
            Rationale string.
        """
        parts = [
            f"Column '{col_analysis.column_name}' is {col_analysis.column_type.value}",
            f"with {col_analysis.missing_pct:.1%} missing values",
            f"({col_analysis.missingness_type.value} mechanism).",
            f"Recommended: {strategy.value}",
        ]

        if strategy == ImputationStrategy.MEDIAN:
            parts.append("- robust to outliers for numeric data.")
        elif strategy == ImputationStrategy.MODE:
            parts.append("- most frequent value for categorical data.")
        elif strategy == ImputationStrategy.KNN:
            parts.append("- uses similar records for conditional imputation.")
        elif strategy == ImputationStrategy.REGRESSION:
            parts.append("- models relationships with predictor columns.")
        elif strategy == ImputationStrategy.MICE:
            parts.append("- iterative multivariate imputation for high missingness.")
        elif strategy == ImputationStrategy.RULE_BASED:
            parts.append("- domain rules needed for MNAR mechanism.")
        elif strategy == ImputationStrategy.LINEAR_INTERPOLATION:
            parts.append("- temporal interpolation for datetime columns.")
        elif strategy == ImputationStrategy.REGULATORY_DEFAULT:
            parts.append("- regulatory default values for very high missingness.")

        return " ".join(parts)

    def _estimate_confidence(
        self,
        strategy: ImputationStrategy,
        missing_pct: float,
        missingness_type: MissingnessType,
    ) -> float:
        """Estimate expected confidence for a strategy and context.

        Higher confidence for MCAR + low missingness + simple methods.
        Lower confidence for MNAR + high missingness.

        Args:
            strategy: Imputation strategy.
            missing_pct: Fraction missing.
            missingness_type: Missingness mechanism.

        Returns:
            Estimated confidence in [0.0, 1.0].
        """
        # Base confidence by strategy
        base_confidence: Dict[ImputationStrategy, float] = {
            ImputationStrategy.MEAN: 0.75,
            ImputationStrategy.MEDIAN: 0.78,
            ImputationStrategy.MODE: 0.72,
            ImputationStrategy.KNN: 0.82,
            ImputationStrategy.REGRESSION: 0.80,
            ImputationStrategy.HOT_DECK: 0.70,
            ImputationStrategy.LOCF: 0.65,
            ImputationStrategy.NOCB: 0.65,
            ImputationStrategy.RANDOM_FOREST: 0.85,
            ImputationStrategy.GRADIENT_BOOSTING: 0.84,
            ImputationStrategy.MICE: 0.88,
            ImputationStrategy.MATRIX_FACTORIZATION: 0.80,
            ImputationStrategy.LINEAR_INTERPOLATION: 0.82,
            ImputationStrategy.SPLINE_INTERPOLATION: 0.78,
            ImputationStrategy.SEASONAL_DECOMPOSITION: 0.80,
            ImputationStrategy.RULE_BASED: 0.90,
            ImputationStrategy.LOOKUP_TABLE: 0.92,
            ImputationStrategy.REGULATORY_DEFAULT: 0.60,
            ImputationStrategy.CUSTOM: 0.75,
        }

        confidence = base_confidence.get(strategy, 0.70)

        # Penalty for high missingness
        if missing_pct > 0.5:
            confidence *= 0.80
        elif missing_pct > 0.3:
            confidence *= 0.90

        # Penalty for MNAR
        if missingness_type == MissingnessType.MNAR:
            confidence *= 0.85
        elif missingness_type == MissingnessType.MAR:
            confidence *= 0.95

        return min(confidence, 1.0)
