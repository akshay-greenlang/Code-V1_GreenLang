# -*- coding: utf-8 -*-
"""
Completeness Analyzer Engine - AGENT-DATA-010: Data Quality Profiler (GL-DATA-X-013)

Null/empty analysis and completeness scoring across datasets. Detects
missing-data patterns (MCAR, MAR, MNAR), computes per-column and
per-record fill rates, identifies gaps in required fields, and generates
actionable quality issues with severity classification.

Zero-Hallucination Guarantees:
    - All completeness scores are deterministic arithmetic (fill_count / total)
    - Missing pattern detection uses statistical heuristics only (stddev, correlation)
    - No ML/LLM calls in the analysis path
    - SHA-256 provenance on every analysis mutation
    - Thread-safe in-memory storage

Missing Pattern Heuristics:
    - MCAR: Uniform missingness (stddev of per-column missing rates < 0.1)
    - MAR: Correlated missingness (when col A null, col B also null > 50%)
    - MNAR: Systematic missingness (> 30% missing in specific columns)
    - UNKNOWN: Default when no pattern detected

Example:
    >>> from greenlang.data_quality_profiler.completeness_analyzer import CompletenessAnalyzer
    >>> analyzer = CompletenessAnalyzer()
    >>> data = [
    ...     {"name": "Alice", "age": 30, "email": None},
    ...     {"name": "Bob", "age": None, "email": "bob@test.com"},
    ... ]
    >>> result = analyzer.analyze(data)
    >>> print(result["completeness_score"], result["missing_pattern"])

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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "CompletenessAnalyzer",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_id(prefix: str = "CMP") -> str:
    """Generate a unique identifier with the given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        String of the form ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash for a completeness operation.

    Args:
        operation: Name of the operation.
        data_repr: Serialised representation of the data involved.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_missing(value: Any) -> bool:
    """Determine whether a value is considered missing/null/empty.

    Args:
        value: The value to check.

    Returns:
        True if the value is None, empty string, or whitespace-only string.
    """
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _safe_stdev(values: List[float]) -> float:
    """Compute sample standard deviation, returning 0.0 for < 2 values.

    Args:
        values: List of numeric values.

    Returns:
        Sample standard deviation or 0.0.
    """
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Missing pattern classifications
PATTERN_MCAR = "MCAR"
PATTERN_MAR = "MAR"
PATTERN_MNAR = "MNAR"
PATTERN_UNKNOWN = "UNKNOWN"

# Severity levels for issues
SEVERITY_CRITICAL = "critical"
SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"
SEVERITY_LOW = "low"
SEVERITY_INFO = "info"

# Thresholds
_MCAR_STDDEV_THRESHOLD = 0.1
_MAR_CORRELATION_THRESHOLD = 0.5
_MNAR_RATE_THRESHOLD = 0.3
_CRITICAL_MISSING_THRESHOLD = 0.5
_HIGH_MISSING_THRESHOLD = 0.3
_MEDIUM_MISSING_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# CompletenessAnalyzer Engine
# ---------------------------------------------------------------------------


class CompletenessAnalyzer:
    """Null/empty analysis and completeness scoring engine.

    Analyses datasets for missing data, computes completeness scores
    across all columns and records, detects missing data patterns
    (MCAR/MAR/MNAR), identifies required-field gaps, and generates
    quality issues with severity classification.

    Thread-safe: all mutations to internal storage are protected by
    a threading lock. SHA-256 provenance hashes on every analysis.

    Attributes:
        _config: Configuration dictionary.
        _lock: Threading lock for thread-safe storage access.
        _analyses: In-memory storage of completed analyses.
        _stats: Aggregate analysis statistics.

    Example:
        >>> analyzer = CompletenessAnalyzer()
        >>> data = [{"a": 1, "b": None}, {"a": None, "b": 2}]
        >>> result = analyzer.analyze(data)
        >>> assert 0.0 <= result["completeness_score"] <= 1.0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CompletenessAnalyzer.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``critical_threshold``: float, missing rate for critical (default 0.5)
                - ``high_threshold``: float, missing rate for high (default 0.3)
                - ``medium_threshold``: float, missing rate for medium (default 0.1)
                - ``mcar_stddev_threshold``: float, stddev for MCAR (default 0.1)
                - ``mar_correlation_threshold``: float, for MAR (default 0.5)
                - ``mnar_rate_threshold``: float, for MNAR (default 0.3)
        """
        self._config = config or {}
        self._critical_threshold: float = self._config.get(
            "critical_threshold", _CRITICAL_MISSING_THRESHOLD
        )
        self._high_threshold: float = self._config.get(
            "high_threshold", _HIGH_MISSING_THRESHOLD
        )
        self._medium_threshold: float = self._config.get(
            "medium_threshold", _MEDIUM_MISSING_THRESHOLD
        )
        self._mcar_stddev: float = self._config.get(
            "mcar_stddev_threshold", _MCAR_STDDEV_THRESHOLD
        )
        self._mar_correlation: float = self._config.get(
            "mar_correlation_threshold", _MAR_CORRELATION_THRESHOLD
        )
        self._mnar_rate: float = self._config.get(
            "mnar_rate_threshold", _MNAR_RATE_THRESHOLD
        )
        self._lock = threading.Lock()
        self._analyses: Dict[str, Dict[str, Any]] = {}
        self._stats: Dict[str, Any] = {
            "analyses_completed": 0,
            "total_rows_analyzed": 0,
            "total_missing_cells": 0,
            "total_analysis_time_ms": 0.0,
        }
        logger.info(
            "CompletenessAnalyzer initialized: critical=%.2f, high=%.2f, medium=%.2f",
            self._critical_threshold, self._high_threshold, self._medium_threshold,
        )

    # ------------------------------------------------------------------
    # Public API - Full Dataset Analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
        required_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyse dataset completeness and return assessment dict.

        Computes per-column completeness, overall score, missing pattern
        detection, required-field gap analysis, and generates issues.

        Args:
            data: List of row dictionaries to analyse.
            columns: Optional subset of columns. If None, uses all keys.
            required_fields: Optional list of fields that must not be null.

        Returns:
            Completeness assessment dict with: analysis_id, completeness_score,
            column_completeness, missing_pattern, record_completeness_stats,
            required_field_gaps, coverage_matrix, issues, provenance_hash.

        Raises:
            ValueError: If data is empty.
        """
        start = time.monotonic()
        if not data:
            raise ValueError("Cannot analyse empty dataset")

        analysis_id = _generate_id("CMP")
        all_keys = columns if columns else list(data[0].keys())
        required = required_fields or []

        # Per-column completeness
        column_results: Dict[str, Dict[str, Any]] = {}
        for col in all_keys:
            values = [row.get(col) for row in data]
            column_results[col] = self.analyze_column(values, col)

        # Overall completeness score
        completeness_score = self.compute_completeness_score(data, all_keys)

        # Missing pattern detection
        missing_pattern = self._detect_missing_pattern_internal(data, all_keys)

        # Required field gaps
        gaps: List[Dict[str, Any]] = []
        if required:
            gaps = self.find_required_gaps(data, required)

        # Coverage matrix
        coverage = self.get_coverage_matrix(data, all_keys)

        # Record completeness stats
        record_scores = [
            self.compute_record_completeness(row, all_keys) for row in data
        ]
        record_stats = self._summarize_record_completeness(record_scores)

        # Issues
        issues = self.generate_completeness_issues(data, all_keys, required)

        # Total missing cells
        total_cells = len(data) * len(all_keys)
        total_missing = sum(
            column_results[col]["null_count"] for col in all_keys
        )

        # Provenance
        provenance_data = json.dumps({
            "analysis_id": analysis_id,
            "row_count": len(data),
            "columns": all_keys,
            "completeness_score": completeness_score,
            "missing_pattern": missing_pattern,
        }, sort_keys=True, default=str)
        provenance_hash = _compute_provenance("analyze", provenance_data)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        result: Dict[str, Any] = {
            "analysis_id": analysis_id,
            "completeness_score": round(completeness_score, 4),
            "missing_pattern": missing_pattern,
            "row_count": len(data),
            "column_count": len(all_keys),
            "total_cells": total_cells,
            "total_missing": total_missing,
            "overall_fill_rate": round(
                1.0 - (total_missing / total_cells) if total_cells > 0 else 0.0, 4
            ),
            "column_completeness": column_results,
            "record_completeness_stats": record_stats,
            "required_field_gaps": gaps,
            "required_field_count": len(required),
            "coverage_matrix": coverage,
            "issues": issues,
            "issue_count": len(issues),
            "provenance_hash": provenance_hash,
            "analysis_time_ms": round(elapsed_ms, 2),
            "created_at": _utcnow().isoformat(),
        }

        # Store and update stats
        with self._lock:
            self._analyses[analysis_id] = result
            self._stats["analyses_completed"] += 1
            self._stats["total_rows_analyzed"] += len(data)
            self._stats["total_missing_cells"] += total_missing
            self._stats["total_analysis_time_ms"] += elapsed_ms

        logger.info(
            "Completeness analysis: id=%s, score=%.4f, pattern=%s, time=%.1fms",
            analysis_id, completeness_score, missing_pattern, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Per-Column Analysis
    # ------------------------------------------------------------------

    def analyze_column(
        self,
        values: List[Any],
        column_name: str,
    ) -> Dict[str, Any]:
        """Analyse completeness of a single column.

        Args:
            values: List of values for this column (may contain None).
            column_name: Name of the column.

        Returns:
            Dict with: column_name, total_count, null_count, null_rate,
            fill_rate, fill_count, severity, provenance_hash.
        """
        total = len(values)
        null_count = sum(1 for v in values if _is_missing(v))
        fill_count = total - null_count
        null_rate = null_count / total if total > 0 else 0.0
        fill_rate = 1.0 - null_rate

        severity = self._classify_severity(null_rate)

        provenance_data = json.dumps({
            "column_name": column_name,
            "total": total,
            "null_count": null_count,
        }, sort_keys=True, default=str)
        provenance_hash = _compute_provenance("analyze_column", provenance_data)

        return {
            "column_name": column_name,
            "total_count": total,
            "null_count": null_count,
            "null_rate": round(null_rate, 4),
            "fill_count": fill_count,
            "fill_rate": round(fill_rate, 4),
            "severity": severity,
            "provenance_hash": provenance_hash,
        }

    # ------------------------------------------------------------------
    # Completeness Score
    # ------------------------------------------------------------------

    def compute_completeness_score(
        self,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
    ) -> float:
        """Compute overall completeness score for a dataset.

        Score is the ratio of non-missing cells to total cells.

        Args:
            data: List of row dictionaries.
            columns: Optional column subset. If None, uses all keys.

        Returns:
            Float between 0.0 and 1.0 representing completeness.
        """
        if not data:
            return 0.0

        cols = columns if columns else list(data[0].keys())
        total_cells = len(data) * len(cols)
        if total_cells == 0:
            return 0.0

        missing = 0
        for row in data:
            for col in cols:
                if _is_missing(row.get(col)):
                    missing += 1

        return (total_cells - missing) / total_cells

    # ------------------------------------------------------------------
    # Missing Pattern Detection
    # ------------------------------------------------------------------

    def detect_missing_pattern(
        self,
        values: List[Any],
    ) -> str:
        """Detect the missing data pattern for a single column.

        This is a simplified single-column version. For full dataset
        pattern detection, use analyze() which considers cross-column
        correlations.

        Args:
            values: List of values (may contain None/empty).

        Returns:
            One of MCAR, MAR, MNAR, UNKNOWN.
        """
        total = len(values)
        if total == 0:
            return PATTERN_UNKNOWN

        missing_count = sum(1 for v in values if _is_missing(v))
        missing_rate = missing_count / total

        if missing_rate > self._mnar_rate:
            return PATTERN_MNAR
        if missing_rate == 0.0:
            return PATTERN_MCAR  # no missing data
        return PATTERN_UNKNOWN

    def _detect_missing_pattern_internal(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
    ) -> str:
        """Detect the missing data pattern across the full dataset.

        Heuristic logic:
        - MCAR: stddev of per-column missing rates < threshold
        - MAR: correlated missingness detected between column pairs
        - MNAR: any column has > threshold missing rate
        - UNKNOWN: default

        Args:
            data: List of row dictionaries.
            columns: Column names to analyse.

        Returns:
            One of MCAR, MAR, MNAR, UNKNOWN.
        """
        if not data or not columns:
            return PATTERN_UNKNOWN

        # Compute per-column missing rates
        missing_rates: Dict[str, float] = {}
        column_missing: Dict[str, List[bool]] = {}
        for col in columns:
            missing_flags = [_is_missing(row.get(col)) for row in data]
            column_missing[col] = missing_flags
            rate = sum(missing_flags) / len(data) if data else 0.0
            missing_rates[col] = rate

        rates_list = list(missing_rates.values())

        # Check if any data is missing at all
        if all(r == 0.0 for r in rates_list):
            return PATTERN_MCAR

        # Check MNAR: any column with high missing rate
        if any(r > self._mnar_rate for r in rates_list):
            # Before concluding MNAR, check for MAR
            if self._check_mar_correlation(column_missing, columns):
                return PATTERN_MAR
            return PATTERN_MNAR

        # Check MCAR: uniform missingness
        rate_stddev = _safe_stdev(rates_list) if len(rates_list) >= 2 else 0.0
        if rate_stddev < self._mcar_stddev:
            return PATTERN_MCAR

        # Check MAR: correlated missingness
        if self._check_mar_correlation(column_missing, columns):
            return PATTERN_MAR

        return PATTERN_UNKNOWN

    def _check_mar_correlation(
        self,
        column_missing: Dict[str, List[bool]],
        columns: List[str],
    ) -> bool:
        """Check for MAR (Missing at Random) correlation between columns.

        MAR is detected when one column being null is correlated with
        another column also being null at a rate above the threshold.

        Args:
            column_missing: Dict mapping column -> list of missing flags.
            columns: Column names.

        Returns:
            True if MAR-like correlation is detected.
        """
        if len(columns) < 2:
            return False

        for i in range(len(columns)):
            col_a = columns[i]
            flags_a = column_missing[col_a]
            a_missing_indices = [idx for idx, f in enumerate(flags_a) if f]

            if not a_missing_indices:
                continue

            for j in range(i + 1, len(columns)):
                col_b = columns[j]
                flags_b = column_missing[col_b]

                # When col A is missing, how often is col B also missing?
                both_missing = sum(1 for idx in a_missing_indices if flags_b[idx])
                correlation = both_missing / len(a_missing_indices) if a_missing_indices else 0.0

                if correlation > self._mar_correlation:
                    return True

        return False

    # ------------------------------------------------------------------
    # Required Field Gaps
    # ------------------------------------------------------------------

    def find_required_gaps(
        self,
        data: List[Dict[str, Any]],
        required_fields: List[str],
    ) -> List[Dict[str, Any]]:
        """Find records where required fields are missing.

        Args:
            data: List of row dictionaries.
            required_fields: List of field names that must not be null.

        Returns:
            List of gap dicts with: row_index, missing_fields, record_id.
        """
        gaps: List[Dict[str, Any]] = []
        for idx, row in enumerate(data):
            missing_in_row: List[str] = []
            for field in required_fields:
                if _is_missing(row.get(field)):
                    missing_in_row.append(field)

            if missing_in_row:
                record_id = row.get("id", row.get("_id", f"row_{idx}"))
                gaps.append({
                    "row_index": idx,
                    "record_id": str(record_id),
                    "missing_fields": missing_in_row,
                    "missing_count": len(missing_in_row),
                    "total_required": len(required_fields),
                    "compliance_rate": round(
                        1.0 - len(missing_in_row) / len(required_fields), 4
                    ) if required_fields else 1.0,
                })

        return gaps

    # ------------------------------------------------------------------
    # Per-Record Completeness
    # ------------------------------------------------------------------

    def compute_record_completeness(
        self,
        record: Dict[str, Any],
        expected_fields: Optional[List[str]] = None,
    ) -> float:
        """Compute completeness score for a single record.

        Args:
            record: A single row dictionary.
            expected_fields: Optional list of expected field names.
                If None, uses all keys in the record.

        Returns:
            Float between 0.0 and 1.0.
        """
        fields = expected_fields if expected_fields else list(record.keys())
        if not fields:
            return 1.0

        filled = sum(1 for f in fields if not _is_missing(record.get(f)))
        return filled / len(fields)

    def _summarize_record_completeness(
        self,
        scores: List[float],
    ) -> Dict[str, Any]:
        """Summarize per-record completeness scores.

        Args:
            scores: List of per-record completeness scores.

        Returns:
            Dict with min, max, mean, median, stddev, fully_complete_count,
            fully_complete_rate.
        """
        if not scores:
            return {
                "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0,
                "stddev": 0.0, "fully_complete_count": 0,
                "fully_complete_rate": 0.0,
            }

        fully_complete = sum(1 for s in scores if s >= 1.0)
        return {
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "mean": round(statistics.mean(scores), 4),
            "median": round(statistics.median(scores), 4),
            "stddev": round(_safe_stdev(scores), 4),
            "fully_complete_count": fully_complete,
            "fully_complete_rate": round(
                fully_complete / len(scores), 4
            ) if scores else 0.0,
            "total_records": len(scores),
        }

    # ------------------------------------------------------------------
    # Coverage Matrix
    # ------------------------------------------------------------------

    def get_coverage_matrix(
        self,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute a coverage matrix: column -> fill_rate.

        Args:
            data: List of row dictionaries.
            columns: Optional column subset. If None, uses all keys.

        Returns:
            Dict mapping column_name -> fill_rate (0.0 to 1.0).
        """
        if not data:
            return {}

        cols = columns if columns else list(data[0].keys())
        total = len(data)
        matrix: Dict[str, float] = {}

        for col in cols:
            filled = sum(1 for row in data if not _is_missing(row.get(col)))
            matrix[col] = round(filled / total, 4) if total > 0 else 0.0

        return matrix

    # ------------------------------------------------------------------
    # Issue Generation
    # ------------------------------------------------------------------

    def generate_completeness_issues(
        self,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
        required_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate completeness quality issues for a dataset.

        Args:
            data: List of row dictionaries.
            columns: Optional column subset.
            required_fields: Optional required field list.

        Returns:
            List of issue dicts with: issue_id, type, severity, column,
            message, details.
        """
        if not data:
            return []

        cols = columns if columns else list(data[0].keys())
        required = required_fields or []
        issues: List[Dict[str, Any]] = []
        total_rows = len(data)

        for col in cols:
            null_count = sum(1 for row in data if _is_missing(row.get(col)))
            null_rate = null_count / total_rows if total_rows > 0 else 0.0

            if null_count == 0:
                continue

            severity = self._classify_severity(null_rate)
            is_required = col in required

            issue: Dict[str, Any] = {
                "issue_id": _generate_id("ISS"),
                "type": "missing_data",
                "severity": severity,
                "column": col,
                "message": (
                    f"Column '{col}' has {null_count}/{total_rows} "
                    f"({null_rate:.1%}) missing values"
                ),
                "details": {
                    "null_count": null_count,
                    "null_rate": round(null_rate, 4),
                    "is_required": is_required,
                },
                "created_at": _utcnow().isoformat(),
            }

            # Escalate severity for required fields
            if is_required and severity in (SEVERITY_LOW, SEVERITY_INFO):
                issue["severity"] = SEVERITY_MEDIUM
                issue["message"] += " (required field)"

            if is_required and null_count > 0:
                issue["severity"] = max(
                    [SEVERITY_CRITICAL, SEVERITY_HIGH, SEVERITY_MEDIUM,
                     SEVERITY_LOW, SEVERITY_INFO].index(severity),
                    [SEVERITY_CRITICAL, SEVERITY_HIGH, SEVERITY_MEDIUM,
                     SEVERITY_LOW, SEVERITY_INFO].index(SEVERITY_HIGH),
                )
                # Ensure at least HIGH for required fields with missing data
                if severity not in (SEVERITY_CRITICAL,):
                    issue["severity"] = SEVERITY_HIGH
                    issue["message"] += " [REQUIRED FIELD]"

            issues.append(issue)

        # Dataset-level issue if overall completeness is low
        total_cells = total_rows * len(cols)
        total_missing = sum(
            1 for row in data for col in cols if _is_missing(row.get(col))
        )
        overall_rate = total_missing / total_cells if total_cells > 0 else 0.0

        if overall_rate > self._medium_threshold:
            issues.append({
                "issue_id": _generate_id("ISS"),
                "type": "low_completeness",
                "severity": self._classify_severity(overall_rate),
                "column": "__dataset__",
                "message": (
                    f"Dataset overall missing rate is {overall_rate:.1%} "
                    f"({total_missing}/{total_cells} cells)"
                ),
                "details": {
                    "total_missing": total_missing,
                    "total_cells": total_cells,
                    "overall_missing_rate": round(overall_rate, 4),
                },
                "created_at": _utcnow().isoformat(),
            })

        return issues

    # ------------------------------------------------------------------
    # Severity Classification
    # ------------------------------------------------------------------

    def _classify_severity(self, missing_rate: float) -> str:
        """Classify severity based on missing rate.

        Args:
            missing_rate: Float between 0.0 and 1.0.

        Returns:
            Severity string.
        """
        if missing_rate >= self._critical_threshold:
            return SEVERITY_CRITICAL
        if missing_rate >= self._high_threshold:
            return SEVERITY_HIGH
        if missing_rate >= self._medium_threshold:
            return SEVERITY_MEDIUM
        if missing_rate > 0.0:
            return SEVERITY_LOW
        return SEVERITY_INFO

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
            limit: Maximum number of results to return.
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
                logger.info("Completeness analysis deleted: %s", analysis_id)
                return True
            return False

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate analysis statistics.

        Returns:
            Dictionary with counters and totals for all completeness
            analyses performed by this engine instance.
        """
        with self._lock:
            completed = self._stats["analyses_completed"]
            avg_time = (
                self._stats["total_analysis_time_ms"] / completed
                if completed > 0 else 0.0
            )
            avg_missing = (
                self._stats["total_missing_cells"] / completed
                if completed > 0 else 0.0
            )
            return {
                "analyses_completed": completed,
                "total_rows_analyzed": self._stats["total_rows_analyzed"],
                "total_missing_cells": self._stats["total_missing_cells"],
                "avg_missing_per_analysis": round(avg_missing, 2),
                "total_analysis_time_ms": round(
                    self._stats["total_analysis_time_ms"], 2
                ),
                "avg_analysis_time_ms": round(avg_time, 2),
                "stored_analyses": len(self._analyses),
                "timestamp": _utcnow().isoformat(),
            }
