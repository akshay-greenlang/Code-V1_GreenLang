# -*- coding: utf-8 -*-
"""
Consistency Analyzer Engine - AGENT-DATA-010: Data Quality Profiler (GL-DATA-X-013)

Format uniformity and cross-column/cross-dataset consistency analysis.
Checks format uniformity (string length coefficient of variation, type
consistency ratio), referential integrity between datasets, schema drift
detection, and distribution comparison using chi-squared and KS-test
approximations.

Zero-Hallucination Guarantees:
    - All uniformity and consistency scores use deterministic arithmetic
    - Distribution comparison uses standard statistical tests (chi-squared, KS)
    - Schema drift detection uses set-based comparison only
    - No ML/LLM calls in the analysis path
    - SHA-256 provenance on every analysis mutation
    - Thread-safe in-memory storage

Example:
    >>> from greenlang.data_quality_profiler.consistency_analyzer import ConsistencyAnalyzer
    >>> analyzer = ConsistencyAnalyzer()
    >>> data = [
    ...     {"status": "active", "score": 85},
    ...     {"status": "ACTIVE", "score": 90},
    ...     {"status": "Active", "score": 78},
    ... ]
    >>> result = analyzer.analyze(data)
    >>> print(result["consistency_score"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-010 Data Quality Profiler (GL-DATA-X-013)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "ConsistencyAnalyzer",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_id(prefix: str = "CST") -> str:
    """Generate a unique identifier with the given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        String of the form ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash for a consistency operation.

    Args:
        operation: Name of the operation.
        data_repr: Serialised representation of the data involved.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_stdev(values: List[float]) -> float:
    """Compute sample standard deviation, 0.0 for < 2 values.

    Args:
        values: List of numeric values.

    Returns:
        Sample standard deviation or 0.0.
    """
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _safe_mean(values: List[float]) -> float:
    """Compute mean, 0.0 for empty lists.

    Args:
        values: List of numeric values.

    Returns:
        Arithmetic mean or 0.0.
    """
    if not values:
        return 0.0
    return statistics.mean(values)


def _classify_value_type(value: Any) -> str:
    """Classify a value into a basic type category.

    Args:
        value: Value to classify.

    Returns:
        Type string: 'null', 'bool', 'int', 'float', 'str'.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITY_CRITICAL = "critical"
SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"
SEVERITY_LOW = "low"
SEVERITY_INFO = "info"

# Drift change types
DRIFT_COLUMN_ADDED = "column_added"
DRIFT_COLUMN_REMOVED = "column_removed"
DRIFT_TYPE_CHANGED = "type_changed"
DRIFT_NULLABLE_CHANGED = "nullable_changed"


# ---------------------------------------------------------------------------
# ConsistencyAnalyzer Engine
# ---------------------------------------------------------------------------


class ConsistencyAnalyzer:
    """Format uniformity and cross-column/cross-dataset consistency engine.

    Analyses datasets for format uniformity within columns, type consistency
    ratios, referential integrity between datasets, schema drift detection,
    and distribution comparison using statistical tests.

    Thread-safe: all mutations to internal storage are protected by
    a threading lock. SHA-256 provenance hashes on every analysis.

    Attributes:
        _config: Configuration dictionary.
        _lock: Threading lock for thread-safe storage access.
        _analyses: In-memory storage of completed analyses.
        _stats: Aggregate analysis statistics.

    Example:
        >>> analyzer = ConsistencyAnalyzer()
        >>> data = [{"a": "hello"}, {"a": "HELLO"}, {"a": "Hello"}]
        >>> result = analyzer.analyze(data)
        >>> assert 0.0 <= result["consistency_score"] <= 1.0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ConsistencyAnalyzer.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``uniformity_weight``: float (default 0.4)
                - ``type_weight``: float (default 0.3)
                - ``value_weight``: float (default 0.3)
                - ``cv_threshold``: float, coefficient of variation
                  threshold for uniformity issues (default 0.5)
        """
        self._config = config or {}
        self._uniformity_weight: float = self._config.get("uniformity_weight", 0.4)
        self._type_weight: float = self._config.get("type_weight", 0.3)
        self._value_weight: float = self._config.get("value_weight", 0.3)
        self._cv_threshold: float = self._config.get("cv_threshold", 0.5)
        self._lock = threading.Lock()
        self._analyses: Dict[str, Dict[str, Any]] = {}
        self._stats: Dict[str, Any] = {
            "analyses_completed": 0,
            "total_rows_analyzed": 0,
            "total_issues_found": 0,
            "total_analysis_time_ms": 0.0,
        }
        logger.info(
            "ConsistencyAnalyzer initialized: weights=%.2f/%.2f/%.2f, cv=%.2f",
            self._uniformity_weight, self._type_weight,
            self._value_weight, self._cv_threshold,
        )

    # ------------------------------------------------------------------
    # Public API - Full Dataset Analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyse dataset consistency and return assessment dict.

        Computes format uniformity, type consistency, and value
        consistency for each column, then aggregates into an overall
        consistency score.

        Args:
            data: List of row dictionaries.
            columns: Optional subset of columns. If None, uses all keys.

        Returns:
            Consistency assessment dict with: analysis_id,
            consistency_score, column_consistency, issues, provenance_hash.

        Raises:
            ValueError: If data is empty.
        """
        start = time.monotonic()
        if not data:
            raise ValueError("Cannot analyse empty dataset")

        analysis_id = _generate_id("CST")
        all_keys = columns if columns else list(data[0].keys())

        # Per-column consistency
        column_results: Dict[str, Dict[str, Any]] = {}
        for col in all_keys:
            values = [row.get(col) for row in data]
            uniformity = self.check_format_uniformity(values, col)
            value_consistency = self.check_value_consistency(values, col)

            # Type consistency ratio
            type_ratio = self._compute_type_consistency(values)

            col_score = (
                uniformity * self._uniformity_weight +
                type_ratio * self._type_weight +
                value_consistency.get("stability_score", 1.0) * self._value_weight
            )

            column_results[col] = {
                "column_name": col,
                "format_uniformity": round(uniformity, 4),
                "type_consistency_ratio": round(type_ratio, 4),
                "value_consistency": value_consistency,
                "column_consistency_score": round(min(max(col_score, 0.0), 1.0), 4),
            }

        # Overall consistency score
        consistency_score = self.compute_consistency_score(data, all_keys, column_results)

        # Issues
        issues = self.generate_consistency_issues(data, all_keys, column_results)

        # Provenance
        provenance_data = json.dumps({
            "analysis_id": analysis_id,
            "row_count": len(data),
            "columns": all_keys,
            "consistency_score": consistency_score,
        }, sort_keys=True, default=str)
        provenance_hash = _compute_provenance("analyze", provenance_data)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        result: Dict[str, Any] = {
            "analysis_id": analysis_id,
            "consistency_score": round(consistency_score, 4),
            "row_count": len(data),
            "column_count": len(all_keys),
            "column_consistency": column_results,
            "issues": issues,
            "issue_count": len(issues),
            "provenance_hash": provenance_hash,
            "analysis_time_ms": round(elapsed_ms, 2),
            "created_at": _utcnow().isoformat(),
        }

        with self._lock:
            self._analyses[analysis_id] = result
            self._stats["analyses_completed"] += 1
            self._stats["total_rows_analyzed"] += len(data)
            self._stats["total_issues_found"] += len(issues)
            self._stats["total_analysis_time_ms"] += elapsed_ms

        logger.info(
            "Consistency analysis: id=%s, score=%.4f, issues=%d, time=%.1fms",
            analysis_id, consistency_score, len(issues), elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Format Uniformity
    # ------------------------------------------------------------------

    def check_format_uniformity(
        self,
        values: List[Any],
        column_name: str,
    ) -> float:
        """Check format uniformity of a column's values.

        Uniformity is computed from:
        1. String length coefficient of variation (lower CV = more uniform)
        2. Type consistency ratio (fraction of values with the dominant type)

        Combined into a score between 0.0 and 1.0.

        Args:
            values: List of values for this column.
            column_name: Column name for logging.

        Returns:
            Uniformity score between 0.0 and 1.0.
        """
        non_null = [v for v in values if v is not None]
        if not non_null:
            return 1.0

        # String length CV
        str_lengths = [len(str(v)) for v in non_null]
        mean_len = _safe_mean([float(l) for l in str_lengths])
        std_len = _safe_stdev([float(l) for l in str_lengths])

        if mean_len > 0:
            cv = std_len / mean_len
            length_uniformity = max(0.0, 1.0 - cv)
        else:
            length_uniformity = 1.0

        # Type consistency ratio
        type_ratio = self._compute_type_consistency(non_null)

        # Weighted combination: 50% length uniformity, 50% type consistency
        uniformity = (length_uniformity * 0.5 + type_ratio * 0.5)
        return min(max(uniformity, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Value Consistency
    # ------------------------------------------------------------------

    def check_value_consistency(
        self,
        values: List[Any],
        column_name: str,
    ) -> Dict[str, Any]:
        """Check value distribution stability for a column.

        Analyses the distribution of values to detect irregularities
        such as extreme dominance of a single value, unexpected uniform
        distributions, or high variability.

        Args:
            values: List of values for this column.
            column_name: Column name.

        Returns:
            Dict with stability_score, dominant_value_ratio,
            distribution_type, entropy.
        """
        non_null = [v for v in values if v is not None]
        if not non_null:
            return {
                "stability_score": 1.0,
                "dominant_value_ratio": 0.0,
                "distribution_type": "empty",
                "entropy": 0.0,
            }

        str_vals = [str(v) for v in non_null]
        counter = Counter(str_vals)
        total = len(str_vals)
        unique_count = len(counter)

        # Dominant value ratio
        most_common_count = counter.most_common(1)[0][1] if counter else 0
        dominant_ratio = most_common_count / total if total > 0 else 0.0

        # Shannon entropy
        entropy = 0.0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        # Normalised entropy (0 = single value, 1 = perfectly uniform)
        max_entropy = math.log2(unique_count) if unique_count > 1 else 1.0
        normalised_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Distribution type classification
        if unique_count == 1:
            dist_type = "constant"
        elif dominant_ratio > 0.9:
            dist_type = "near_constant"
        elif normalised_entropy > 0.95:
            dist_type = "uniform"
        elif normalised_entropy > 0.5:
            dist_type = "balanced"
        else:
            dist_type = "skewed"

        # Stability score: penalise near-constant or heavily skewed
        if dist_type == "constant":
            stability_score = 1.0
        elif dist_type == "near_constant":
            stability_score = 0.95
        elif dist_type == "uniform":
            stability_score = 0.9
        elif dist_type == "balanced":
            stability_score = 0.85
        else:
            stability_score = 0.7

        return {
            "stability_score": round(stability_score, 4),
            "dominant_value_ratio": round(dominant_ratio, 4),
            "distribution_type": dist_type,
            "entropy": round(entropy, 4),
            "normalised_entropy": round(normalised_entropy, 4),
            "unique_count": unique_count,
        }

    # ------------------------------------------------------------------
    # Referential Integrity
    # ------------------------------------------------------------------

    def check_referential_integrity(
        self,
        data: List[Dict[str, Any]],
        foreign_key: str,
        reference_data: List[Dict[str, Any]],
        primary_key: str,
    ) -> Dict[str, Any]:
        """Check referential integrity between two datasets.

        Verifies that every foreign key value in the data exists as a
        primary key value in the reference dataset.

        Args:
            data: List of row dictionaries (child table).
            foreign_key: Column name in data containing foreign key values.
            reference_data: List of row dicts (parent table).
            primary_key: Column name in reference_data with primary keys.

        Returns:
            Dict with: total_refs, matched_refs, orphaned_refs,
            integrity_ratio, orphaned_values.
        """
        # Build primary key set
        pk_set: Set[str] = set()
        for row in reference_data:
            pk_val = row.get(primary_key)
            if pk_val is not None:
                pk_set.add(str(pk_val))

        # Check foreign keys
        total_refs = 0
        matched = 0
        orphaned_values: List[str] = []

        for row in data:
            fk_val = row.get(foreign_key)
            if fk_val is None:
                continue
            total_refs += 1
            fk_str = str(fk_val)
            if fk_str in pk_set:
                matched += 1
            else:
                if len(orphaned_values) < 100:
                    orphaned_values.append(fk_str)

        orphaned = total_refs - matched
        integrity_ratio = matched / total_refs if total_refs > 0 else 1.0

        provenance_data = json.dumps({
            "foreign_key": foreign_key,
            "primary_key": primary_key,
            "total_refs": total_refs,
            "matched": matched,
        }, sort_keys=True, default=str)
        provenance_hash = _compute_provenance("referential_integrity", provenance_data)

        return {
            "foreign_key": foreign_key,
            "primary_key": primary_key,
            "total_refs": total_refs,
            "matched_refs": matched,
            "orphaned_refs": orphaned,
            "integrity_ratio": round(integrity_ratio, 4),
            "orphaned_values": orphaned_values[:20],
            "provenance_hash": provenance_hash,
        }

    # ------------------------------------------------------------------
    # Schema Drift Detection
    # ------------------------------------------------------------------

    def detect_schema_drift(
        self,
        current_schema: Dict[str, str],
        baseline_schema: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Detect schema drift between current and baseline schemas.

        Compares column names and types to identify additions, removals,
        and type changes.

        Args:
            current_schema: Dict mapping column_name -> type_name (current).
            baseline_schema: Dict mapping column_name -> type_name (baseline).

        Returns:
            List of drift item dicts with: column, change_type, details.
        """
        drifts: List[Dict[str, Any]] = []
        current_cols = set(current_schema.keys())
        baseline_cols = set(baseline_schema.keys())

        # Columns added
        for col in sorted(current_cols - baseline_cols):
            drifts.append({
                "column": col,
                "change_type": DRIFT_COLUMN_ADDED,
                "details": {
                    "new_type": current_schema[col],
                },
            })

        # Columns removed
        for col in sorted(baseline_cols - current_cols):
            drifts.append({
                "column": col,
                "change_type": DRIFT_COLUMN_REMOVED,
                "details": {
                    "old_type": baseline_schema[col],
                },
            })

        # Type changes
        for col in sorted(current_cols & baseline_cols):
            if current_schema[col] != baseline_schema[col]:
                drifts.append({
                    "column": col,
                    "change_type": DRIFT_TYPE_CHANGED,
                    "details": {
                        "old_type": baseline_schema[col],
                        "new_type": current_schema[col],
                    },
                })

        if drifts:
            provenance_data = json.dumps({
                "drift_count": len(drifts),
                "current_cols": len(current_cols),
                "baseline_cols": len(baseline_cols),
            }, sort_keys=True, default=str)
            provenance_hash = _compute_provenance("schema_drift", provenance_data)
            for d in drifts:
                d["provenance_hash"] = provenance_hash

        return drifts

    # ------------------------------------------------------------------
    # Distribution Comparison
    # ------------------------------------------------------------------

    def compare_distributions(
        self,
        values_a: List[Any],
        values_b: List[Any],
    ) -> float:
        """Compare two value distributions and return a similarity score.

        Uses a chi-squared test approximation for categorical data or
        a Kolmogorov-Smirnov approximation for numeric data.

        Args:
            values_a: First list of values.
            values_b: Second list of values.

        Returns:
            Similarity score between 0.0 (completely different) and
            1.0 (identical distribution).
        """
        if not values_a or not values_b:
            return 0.0

        # Determine if numeric
        is_numeric = self._are_numeric(values_a) and self._are_numeric(values_b)

        if is_numeric:
            return self._ks_similarity(values_a, values_b)
        else:
            return self._chi_squared_similarity(values_a, values_b)

    def _ks_similarity(
        self,
        values_a: List[Any],
        values_b: List[Any],
    ) -> float:
        """Kolmogorov-Smirnov test approximation for numeric distributions.

        Args:
            values_a: First numeric list.
            values_b: Second numeric list.

        Returns:
            Similarity score (1 - KS statistic).
        """
        nums_a = sorted(float(v) for v in values_a if v is not None)
        nums_b = sorted(float(v) for v in values_b if v is not None)

        if not nums_a or not nums_b:
            return 0.0

        n_a = len(nums_a)
        n_b = len(nums_b)

        # Merge and compute ECDF difference
        all_vals = sorted(set(nums_a + nums_b))
        max_diff = 0.0

        for val in all_vals:
            # ECDF for A
            ecdf_a = sum(1 for x in nums_a if x <= val) / n_a
            # ECDF for B
            ecdf_b = sum(1 for x in nums_b if x <= val) / n_b
            diff = abs(ecdf_a - ecdf_b)
            if diff > max_diff:
                max_diff = diff

        return round(max(0.0, 1.0 - max_diff), 4)

    def _chi_squared_similarity(
        self,
        values_a: List[Any],
        values_b: List[Any],
    ) -> float:
        """Chi-squared test approximation for categorical distributions.

        Args:
            values_a: First categorical list.
            values_b: Second categorical list.

        Returns:
            Similarity score based on normalised chi-squared statistic.
        """
        str_a = [str(v) for v in values_a if v is not None]
        str_b = [str(v) for v in values_b if v is not None]

        if not str_a or not str_b:
            return 0.0

        counter_a = Counter(str_a)
        counter_b = Counter(str_b)
        all_categories = set(counter_a.keys()) | set(counter_b.keys())

        total_a = len(str_a)
        total_b = len(str_b)

        # Compute chi-squared statistic
        chi_sq = 0.0
        for cat in all_categories:
            # Observed proportions
            p_a = counter_a.get(cat, 0) / total_a
            p_b = counter_b.get(cat, 0) / total_b
            # Average expected proportion
            expected = (p_a + p_b) / 2
            if expected > 0:
                chi_sq += ((p_a - p_b) ** 2) / expected

        # Normalise to [0, 1] similarity
        # Max chi_sq for perfectly different distributions is 2.0
        normalised = min(chi_sq / 2.0, 1.0)
        return round(max(0.0, 1.0 - normalised), 4)

    def _are_numeric(self, values: List[Any]) -> bool:
        """Check if the majority of values are numeric.

        Args:
            values: List of values to check.

        Returns:
            True if >80% of non-null values are numeric.
        """
        non_null = [v for v in values if v is not None]
        if not non_null:
            return False

        numeric_count = 0
        for v in non_null:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_count += 1
            else:
                try:
                    float(str(v))
                    numeric_count += 1
                except (ValueError, TypeError):
                    pass

        return (numeric_count / len(non_null)) > 0.8

    # ------------------------------------------------------------------
    # Cross-Dataset Comparison
    # ------------------------------------------------------------------

    def compare_datasets(
        self,
        data_a: List[Dict[str, Any]],
        data_b: List[Dict[str, Any]],
        key_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare two datasets for consistency.

        Compares schema overlap, distribution similarity per column,
        and key-based record matching.

        Args:
            data_a: First dataset.
            data_b: Second dataset.
            key_columns: Optional columns to use as matching keys.

        Returns:
            Comparison dict with: schema_overlap, column_similarities,
            key_match_stats, overall_similarity.
        """
        if not data_a or not data_b:
            return {
                "schema_overlap": 0.0,
                "column_similarities": {},
                "key_match_stats": {},
                "overall_similarity": 0.0,
            }

        cols_a = set(data_a[0].keys())
        cols_b = set(data_b[0].keys())
        common_cols = cols_a & cols_b

        # Schema overlap
        all_cols = cols_a | cols_b
        schema_overlap = len(common_cols) / len(all_cols) if all_cols else 1.0

        # Per-column distribution similarity
        col_similarities: Dict[str, float] = {}
        for col in sorted(common_cols):
            vals_a = [row.get(col) for row in data_a if row.get(col) is not None]
            vals_b = [row.get(col) for row in data_b if row.get(col) is not None]
            if vals_a and vals_b:
                col_similarities[col] = self.compare_distributions(vals_a, vals_b)
            else:
                col_similarities[col] = 0.0

        # Key-based matching
        key_stats: Dict[str, Any] = {}
        if key_columns:
            key_stats = self._compare_keys(data_a, data_b, key_columns)

        # Overall similarity
        if col_similarities:
            avg_sim = sum(col_similarities.values()) / len(col_similarities)
        else:
            avg_sim = 0.0

        overall = (schema_overlap * 0.3 + avg_sim * 0.7)

        provenance_data = json.dumps({
            "rows_a": len(data_a),
            "rows_b": len(data_b),
            "common_cols": len(common_cols),
            "overall_similarity": overall,
        }, sort_keys=True, default=str)
        provenance_hash = _compute_provenance("compare_datasets", provenance_data)

        return {
            "schema_overlap": round(schema_overlap, 4),
            "common_columns": sorted(common_cols),
            "columns_only_in_a": sorted(cols_a - cols_b),
            "columns_only_in_b": sorted(cols_b - cols_a),
            "column_similarities": col_similarities,
            "key_match_stats": key_stats,
            "overall_similarity": round(overall, 4),
            "provenance_hash": provenance_hash,
        }

    def _compare_keys(
        self,
        data_a: List[Dict[str, Any]],
        data_b: List[Dict[str, Any]],
        key_columns: List[str],
    ) -> Dict[str, Any]:
        """Compare datasets by key columns.

        Args:
            data_a: First dataset.
            data_b: Second dataset.
            key_columns: Columns forming the composite key.

        Returns:
            Dict with matched, only_in_a, only_in_b, match_rate.
        """
        def make_key(row: Dict[str, Any]) -> str:
            parts = [str(row.get(k, "")) for k in key_columns]
            return "|".join(parts)

        keys_a = set(make_key(row) for row in data_a)
        keys_b = set(make_key(row) for row in data_b)

        matched = keys_a & keys_b
        only_a = keys_a - keys_b
        only_b = keys_b - keys_a
        total = len(keys_a | keys_b)
        match_rate = len(matched) / total if total > 0 else 0.0

        return {
            "matched_count": len(matched),
            "only_in_a_count": len(only_a),
            "only_in_b_count": len(only_b),
            "total_unique_keys": total,
            "match_rate": round(match_rate, 4),
        }

    # ------------------------------------------------------------------
    # Consistency Score
    # ------------------------------------------------------------------

    def compute_consistency_score(
        self,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
        column_results: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> float:
        """Compute overall consistency score for a dataset.

        If column_results are provided, averages per-column scores.
        Otherwise computes from scratch.

        Args:
            data: List of row dictionaries.
            columns: Optional column subset.
            column_results: Optional pre-computed column results.

        Returns:
            Float between 0.0 and 1.0.
        """
        if column_results:
            scores = [
                cr.get("column_consistency_score", 1.0)
                for cr in column_results.values()
            ]
            if scores:
                return sum(scores) / len(scores)
            return 1.0

        if not data:
            return 1.0

        cols = columns if columns else list(data[0].keys())
        total_score = 0.0
        col_count = 0

        for col in cols:
            values = [row.get(col) for row in data]
            uniformity = self.check_format_uniformity(values, col)
            type_ratio = self._compute_type_consistency(values)
            value_cons = self.check_value_consistency(values, col)

            col_score = (
                uniformity * self._uniformity_weight +
                type_ratio * self._type_weight +
                value_cons.get("stability_score", 1.0) * self._value_weight
            )
            total_score += min(max(col_score, 0.0), 1.0)
            col_count += 1

        return total_score / col_count if col_count > 0 else 1.0

    # ------------------------------------------------------------------
    # Type Consistency
    # ------------------------------------------------------------------

    def _compute_type_consistency(self, values: List[Any]) -> float:
        """Compute type consistency ratio for a list of values.

        Returns the fraction of non-null values that share the
        dominant type.

        Args:
            values: List of values.

        Returns:
            Float between 0.0 and 1.0.
        """
        non_null = [v for v in values if v is not None]
        if not non_null:
            return 1.0

        type_counts: Dict[str, int] = {}
        for v in non_null:
            t = _classify_value_type(v)
            type_counts[t] = type_counts.get(t, 0) + 1

        dominant_count = max(type_counts.values())
        return dominant_count / len(non_null)

    # ------------------------------------------------------------------
    # Issue Generation
    # ------------------------------------------------------------------

    def generate_consistency_issues(
        self,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
        column_results: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate consistency quality issues for a dataset.

        Args:
            data: List of row dictionaries.
            columns: Optional column subset.
            column_results: Optional pre-computed column results.

        Returns:
            List of issue dicts with: issue_id, type, severity,
            column, message, details.
        """
        issues: List[Dict[str, Any]] = []

        if column_results is None:
            return issues

        for col_name, col_result in column_results.items():
            # Low format uniformity
            uniformity = col_result.get("format_uniformity", 1.0)
            if uniformity < 0.7:
                severity = SEVERITY_HIGH if uniformity < 0.5 else SEVERITY_MEDIUM
                issues.append({
                    "issue_id": _generate_id("ISS"),
                    "type": "low_format_uniformity",
                    "severity": severity,
                    "column": col_name,
                    "message": (
                        f"Column '{col_name}' has low format uniformity "
                        f"({uniformity:.1%})"
                    ),
                    "details": {
                        "format_uniformity": uniformity,
                    },
                    "created_at": _utcnow().isoformat(),
                })

            # Low type consistency
            type_ratio = col_result.get("type_consistency_ratio", 1.0)
            if type_ratio < 0.9:
                severity = SEVERITY_HIGH if type_ratio < 0.7 else SEVERITY_MEDIUM
                issues.append({
                    "issue_id": _generate_id("ISS"),
                    "type": "mixed_types",
                    "severity": severity,
                    "column": col_name,
                    "message": (
                        f"Column '{col_name}' has mixed data types "
                        f"(consistency {type_ratio:.1%})"
                    ),
                    "details": {
                        "type_consistency_ratio": type_ratio,
                    },
                    "created_at": _utcnow().isoformat(),
                })

            # Case inconsistency (check for string columns)
            value_cons = col_result.get("value_consistency", {})
            dist_type = value_cons.get("distribution_type", "balanced")
            if dist_type == "skewed":
                issues.append({
                    "issue_id": _generate_id("ISS"),
                    "type": "skewed_distribution",
                    "severity": SEVERITY_LOW,
                    "column": col_name,
                    "message": (
                        f"Column '{col_name}' has a skewed value distribution"
                    ),
                    "details": value_cons,
                    "created_at": _utcnow().isoformat(),
                })

        return issues

    # ------------------------------------------------------------------
    # Storage and Retrieval
    # ------------------------------------------------------------------

    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored analysis by ID.

        Args:
            analysis_id: The analysis identifier.

        Returns:
            Analysis dict or None if not found.
        """
        with self._lock:
            return self._analyses.get(analysis_id)

    def list_analyses(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List stored analyses with pagination.

        Args:
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of analysis dicts sorted by creation time descending.
        """
        with self._lock:
            all_analyses = sorted(
                self._analyses.values(),
                key=lambda a: a.get("created_at", ""),
                reverse=True,
            )
            return all_analyses[offset:offset + limit]

    def delete_analysis(self, analysis_id: str) -> bool:
        """Delete a stored analysis.

        Args:
            analysis_id: The analysis identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if analysis_id in self._analyses:
                del self._analyses[analysis_id]
                logger.info("Consistency analysis deleted: %s", analysis_id)
                return True
            return False

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate analysis statistics.

        Returns:
            Dictionary with counters and totals for all consistency
            analyses performed by this engine instance.
        """
        with self._lock:
            completed = self._stats["analyses_completed"]
            avg_time = (
                self._stats["total_analysis_time_ms"] / completed
                if completed > 0 else 0.0
            )
            avg_issues = (
                self._stats["total_issues_found"] / completed
                if completed > 0 else 0.0
            )
            return {
                "analyses_completed": completed,
                "total_rows_analyzed": self._stats["total_rows_analyzed"],
                "total_issues_found": self._stats["total_issues_found"],
                "avg_issues_per_analysis": round(avg_issues, 2),
                "total_analysis_time_ms": round(
                    self._stats["total_analysis_time_ms"], 2
                ),
                "avg_analysis_time_ms": round(avg_time, 2),
                "stored_analyses": len(self._analyses),
                "timestamp": _utcnow().isoformat(),
            }
