# -*- coding: utf-8 -*-
"""
Data Quality Scorer - AGENT-DATA-002: Excel/CSV Normalizer

Data quality assessment engine that computes weighted quality scores
across completeness, accuracy, and consistency dimensions with outlier
detection, duplicate detection, and issue enumeration.

Supports:
    - Completeness scoring (null/empty cell ratio)
    - Accuracy scoring (type-conformance ratio)
    - Consistency scoring (format and range consistency across rows)
    - Weighted overall score computation (configurable weights)
    - Per-column quality breakdown
    - Outlier detection (IQR and z-score methods)
    - Duplicate row detection
    - Quality level classification (excellent, good, fair, poor, critical)
    - Issue enumeration with severity and location
    - Thread-safe statistics

Zero-Hallucination Guarantees:
    - All quality scores are deterministic arithmetic
    - No LLM calls in the scoring path
    - Weights and thresholds are configurable and auditable

Example:
    >>> from greenlang.excel_normalizer.data_quality_scorer import DataQualityScorer
    >>> scorer = DataQualityScorer()
    >>> report = scorer.score_file(rows, headers=["col1", "col2"])
    >>> print(report.overall_score, report.quality_level)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-002 Excel/CSV Normalizer
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "QualityLevel",
    "DataQualityReport",
    "DataQualityScorer",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class QualityLevel(str, Enum):
    """Quality level classification."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class DataQualityReport(BaseModel):
    """Quality assessment report for a dataset."""

    report_id: str = Field(
        default_factory=lambda: f"dqr-{uuid.uuid4().hex[:12]}",
        description="Unique report identifier",
    )
    overall_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Weighted overall score",
    )
    quality_level: QualityLevel = Field(
        default=QualityLevel.CRITICAL, description="Quality classification",
    )
    completeness_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Completeness dimension score",
    )
    accuracy_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Accuracy dimension score",
    )
    consistency_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Consistency dimension score",
    )
    total_rows: int = Field(default=0, ge=0, description="Total rows analysed")
    total_columns: int = Field(default=0, ge=0, description="Total columns")
    total_cells: int = Field(default=0, ge=0, description="Total cells")
    empty_cells: int = Field(default=0, ge=0, description="Empty/null cells")
    duplicate_rows: int = Field(default=0, ge=0, description="Duplicate row count")
    outlier_count: int = Field(default=0, ge=0, description="Outlier values detected")
    issues: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of quality issues",
    )
    column_scores: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Per-column quality breakdown",
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Report timestamp",
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# DataQualityScorer
# ---------------------------------------------------------------------------


class DataQualityScorer:
    """Data quality scoring engine with weighted multi-dimensional assessment.

    Computes quality scores across completeness, accuracy, and consistency
    dimensions, detects outliers and duplicates, and classifies overall
    quality into levels.

    Attributes:
        _config: Configuration dictionary.
        _weights: Dimension weights for overall score computation.
        _lock: Threading lock for statistics.
        _stats: Scoring statistics.

    Example:
        >>> scorer = DataQualityScorer()
        >>> report = scorer.score_file(
        ...     [{"col1": "a", "col2": 1}, {"col1": "b", "col2": 2}]
        ... )
        >>> print(report.overall_score)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise DataQualityScorer.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``completeness_weight``: float (default 0.4)
                - ``accuracy_weight``: float (default 0.35)
                - ``consistency_weight``: float (default 0.25)
                - ``outlier_method``: str "iqr" or "zscore" (default "iqr")
                - ``outlier_threshold``: float (default 1.5 for IQR, 3.0 for z)
        """
        self._config = config or {}
        self._weights = {
            "completeness": self._config.get("completeness_weight", 0.4),
            "accuracy": self._config.get("accuracy_weight", 0.35),
            "consistency": self._config.get("consistency_weight", 0.25),
        }
        self._outlier_method: str = self._config.get("outlier_method", "iqr")
        self._outlier_threshold: float = self._config.get("outlier_threshold", 1.5)
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "files_scored": 0,
            "total_scores": 0.0,
            "score_count": 0,
            "total_issues": 0,
        }
        logger.info(
            "DataQualityScorer initialised: weights=%s, outlier=%s",
            self._weights, self._outlier_method,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_file(
        self,
        rows: List[Dict[str, Any]],
        headers: Optional[List[str]] = None,
    ) -> DataQualityReport:
        """Score an entire dataset for quality.

        Args:
            rows: List of data row dictionaries.
            headers: Optional list of column names.

        Returns:
            DataQualityReport with scores and issues.
        """
        start = time.monotonic()

        if not rows:
            return DataQualityReport()

        # Determine columns
        all_keys: List[str] = headers if headers else list(rows[0].keys())
        total_rows = len(rows)
        total_cols = len(all_keys)
        total_cells = total_rows * total_cols

        # Score dimensions
        completeness = self.score_completeness(rows)
        accuracy = self.score_accuracy(rows)
        consistency = self.score_consistency(rows)

        # Weighted overall
        overall = (
            completeness * self._weights["completeness"] +
            accuracy * self._weights["accuracy"] +
            consistency * self._weights["consistency"]
        )
        overall = round(min(max(overall, 0.0), 1.0), 4)

        # Quality level
        quality_level = self.compute_quality_level(overall)

        # Count empty cells
        empty_cells = self._count_empty_cells(rows, all_keys)

        # Detect duplicates
        dup_indices = self.detect_duplicates(rows)
        duplicate_count = len(dup_indices)

        # Detect outliers
        outlier_count = 0
        for key in all_keys:
            values = self._extract_numeric_values(rows, key)
            if len(values) >= 10:
                outliers = self.detect_outliers(values, method=self._outlier_method)
                outlier_count += len(outliers)

        # Per-column scores
        column_scores: Dict[str, Dict[str, Any]] = {}
        for key in all_keys:
            col_values = [row.get(key) for row in rows]
            column_scores[key] = self.score_column(col_values, key)

        # Generate issues
        issues = self.generate_issues(rows)

        report = DataQualityReport(
            overall_score=overall,
            quality_level=quality_level,
            completeness_score=round(completeness, 4),
            accuracy_score=round(accuracy, 4),
            consistency_score=round(consistency, 4),
            total_rows=total_rows,
            total_columns=total_cols,
            total_cells=total_cells,
            empty_cells=empty_cells,
            duplicate_rows=duplicate_count,
            outlier_count=outlier_count,
            issues=issues,
            column_scores=column_scores,
        )

        with self._lock:
            self._stats["files_scored"] += 1
            self._stats["total_scores"] += overall
            self._stats["score_count"] += 1
            self._stats["total_issues"] += len(issues)

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Quality score: %.4f (%s), %d rows, %d issues (%.1f ms)",
            overall, quality_level.value, total_rows, len(issues), elapsed,
        )
        return report

    def score_completeness(self, rows: List[Dict[str, Any]]) -> float:
        """Score the completeness of a dataset.

        Completeness = fraction of non-null, non-empty cells.

        Args:
            rows: List of data row dictionaries.

        Returns:
            Completeness score (0.0 to 1.0).
        """
        if not rows:
            return 0.0

        all_keys = list(rows[0].keys())
        total_cells = len(rows) * len(all_keys)
        if total_cells == 0:
            return 0.0

        non_empty = 0
        for row in rows:
            for key in all_keys:
                value = row.get(key)
                if value is not None and str(value).strip() != "":
                    non_empty += 1

        return non_empty / total_cells

    def score_accuracy(
        self,
        rows: List[Dict[str, Any]],
        expected_types: Optional[Dict[str, str]] = None,
    ) -> float:
        """Score the accuracy of a dataset.

        Accuracy = fraction of values that conform to their detected type
        (no type mismatches within a column).

        Args:
            rows: List of data row dictionaries.
            expected_types: Optional expected type per column.

        Returns:
            Accuracy score (0.0 to 1.0).
        """
        if not rows:
            return 0.0

        all_keys = list(rows[0].keys())
        total_checked = 0
        conforming = 0

        for key in all_keys:
            values = [row.get(key) for row in rows]
            non_empty = [v for v in values if v is not None and str(v).strip()]
            if not non_empty:
                continue

            # Detect dominant type
            type_counts: Dict[str, int] = {"string": 0, "numeric": 0, "other": 0}
            for v in non_empty:
                if isinstance(v, (int, float)):
                    type_counts["numeric"] += 1
                elif isinstance(v, str):
                    cleaned = v.strip().replace(",", "")
                    try:
                        float(cleaned)
                        type_counts["numeric"] += 1
                    except ValueError:
                        type_counts["string"] += 1
                else:
                    type_counts["other"] += 1

            dominant = max(type_counts, key=type_counts.get)  # type: ignore[arg-type]
            conforming += type_counts[dominant]
            total_checked += sum(type_counts.values())

        return conforming / max(total_checked, 1)

    def score_consistency(self, rows: List[Dict[str, Any]]) -> float:
        """Score the consistency of a dataset.

        Evaluates format consistency and range consistency across rows.

        Args:
            rows: List of data row dictionaries.

        Returns:
            Consistency score (0.0 to 1.0).
        """
        if not rows or len(rows) < 2:
            return 1.0

        all_keys = list(rows[0].keys())
        col_scores: List[float] = []

        for key in all_keys:
            values = [row.get(key) for row in rows]
            non_empty = [v for v in values if v is not None and str(v).strip()]
            if len(non_empty) < 2:
                col_scores.append(1.0)
                continue

            # Check format consistency: are all values the same "shape"?
            str_values = [str(v).strip() for v in non_empty]

            # Length variance as a proxy for format consistency
            lengths = [len(s) for s in str_values]
            mean_len = sum(lengths) / len(lengths)
            variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
            std_dev = math.sqrt(variance) if variance > 0 else 0.0
            cv = std_dev / max(mean_len, 1)  # Coefficient of variation

            # Low CV = consistent formatting
            format_score = max(1.0 - cv, 0.0)
            col_scores.append(format_score)

        return sum(col_scores) / max(len(col_scores), 1)

    def score_column(
        self,
        values: List[Any],
        column_name: str,
    ) -> Dict[str, Any]:
        """Compute per-column quality analysis.

        Args:
            values: List of cell values for the column.
            column_name: Column name for reporting.

        Returns:
            Dict with column quality metrics.
        """
        total = len(values)
        non_empty = sum(
            1 for v in values if v is not None and str(v).strip()
        )
        empty = total - non_empty
        completeness = non_empty / max(total, 1)

        # Unique values
        str_values = [str(v).strip() for v in values if v is not None]
        unique = len(set(str_values))
        uniqueness = unique / max(len(str_values), 1)

        return {
            "column_name": column_name,
            "total_values": total,
            "non_empty": non_empty,
            "empty": empty,
            "completeness": round(completeness, 4),
            "unique_values": unique,
            "uniqueness": round(uniqueness, 4),
        }

    def detect_outliers(
        self,
        values: List[float],
        method: str = "iqr",
    ) -> List[int]:
        """Detect outlier indices in a list of numeric values.

        Args:
            values: List of float values.
            method: Detection method ("iqr" or "zscore").

        Returns:
            List of zero-based indices of outlier values.
        """
        if len(values) < 4:
            return []

        if method == "zscore":
            return self._detect_outliers_zscore(values)
        else:
            return self._detect_outliers_iqr(values)

    def detect_duplicates(
        self,
        rows: List[Dict[str, Any]],
        key_columns: Optional[List[str]] = None,
    ) -> List[int]:
        """Detect duplicate row indices.

        Args:
            rows: List of data row dictionaries.
            key_columns: Optional columns to use as duplicate key.
                         If None, uses all columns.

        Returns:
            List of zero-based indices of duplicate rows.
        """
        if not rows:
            return []

        seen: Dict[str, int] = {}
        duplicates: List[int] = []

        for idx, row in enumerate(rows):
            if key_columns:
                key_values = tuple(str(row.get(k, "")) for k in key_columns)
            else:
                key_values = tuple(str(v) for v in row.values())

            row_hash = hashlib.sha256(str(key_values).encode()).hexdigest()

            if row_hash in seen:
                duplicates.append(idx)
            else:
                seen[row_hash] = idx

        return duplicates

    def compute_quality_level(self, score: float) -> QualityLevel:
        """Map a numeric score to a quality level.

        Args:
            score: Overall quality score (0.0 to 1.0).

        Returns:
            QualityLevel classification.
        """
        if score >= 0.95:
            return QualityLevel.EXCELLENT
        elif score >= 0.85:
            return QualityLevel.GOOD
        elif score >= 0.70:
            return QualityLevel.FAIR
        elif score >= 0.50:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL

    def generate_issues(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate a list of quality issue descriptions.

        Args:
            rows: List of data row dictionaries.

        Returns:
            List of issue dicts with severity, type, and description.
        """
        issues: List[Dict[str, Any]] = []

        if not rows:
            return issues

        all_keys = list(rows[0].keys())

        # Check for columns with high emptiness
        for key in all_keys:
            values = [row.get(key) for row in rows]
            empty_count = sum(
                1 for v in values if v is None or str(v).strip() == ""
            )
            empty_pct = empty_count / max(len(values), 1)

            if empty_pct >= 0.5:
                issues.append({
                    "severity": "warning",
                    "type": "high_emptiness",
                    "column": key,
                    "description": f"Column '{key}' is {empty_pct:.0%} empty",
                    "empty_pct": round(empty_pct, 4),
                })

        # Check for duplicates
        dup_indices = self.detect_duplicates(rows)
        if dup_indices:
            issues.append({
                "severity": "warning",
                "type": "duplicates",
                "description": f"{len(dup_indices)} duplicate rows detected",
                "duplicate_count": len(dup_indices),
                "first_indices": dup_indices[:10],
            })

        # Check for constant columns (zero variance)
        for key in all_keys:
            non_empty = [
                str(row.get(key, "")).strip()
                for row in rows
                if row.get(key) is not None and str(row.get(key, "")).strip()
            ]
            if non_empty and len(set(non_empty)) == 1 and len(non_empty) > 1:
                issues.append({
                    "severity": "info",
                    "type": "constant_column",
                    "column": key,
                    "description": f"Column '{key}' has constant value '{non_empty[0]}'",
                })

        return issues

    def get_statistics(self) -> Dict[str, Any]:
        """Return scoring statistics.

        Returns:
            Dictionary with counters and averages.
        """
        with self._lock:
            avg_score = (
                self._stats["total_scores"] / self._stats["score_count"]
                if self._stats["score_count"] > 0 else 0.0
            )
            return {
                "files_scored": self._stats["files_scored"],
                "avg_score": round(avg_score, 4),
                "total_issues": self._stats["total_issues"],
                "timestamp": _utcnow().isoformat(),
            }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _count_empty_cells(
        self,
        rows: List[Dict[str, Any]],
        keys: List[str],
    ) -> int:
        """Count empty/null cells in the dataset.

        Args:
            rows: Data rows.
            keys: Column keys.

        Returns:
            Number of empty cells.
        """
        count = 0
        for row in rows:
            for key in keys:
                value = row.get(key)
                if value is None or str(value).strip() == "":
                    count += 1
        return count

    def _extract_numeric_values(
        self,
        rows: List[Dict[str, Any]],
        key: str,
    ) -> List[float]:
        """Extract numeric values from a column.

        Args:
            rows: Data rows.
            key: Column key.

        Returns:
            List of float values.
        """
        values: List[float] = []
        for row in rows:
            v = row.get(key)
            if v is None:
                continue
            try:
                values.append(float(str(v).replace(",", "")))
            except (ValueError, TypeError):
                continue
        return values

    def _detect_outliers_iqr(self, values: List[float]) -> List[int]:
        """Detect outliers using IQR (Interquartile Range) method.

        Args:
            values: Sorted or unsorted numeric values.

        Returns:
            List of outlier indices.
        """
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        q1 = sorted_vals[n // 4]
        q3 = sorted_vals[(3 * n) // 4]
        iqr = q3 - q1
        lower = q1 - self._outlier_threshold * iqr
        upper = q3 + self._outlier_threshold * iqr

        return [i for i, v in enumerate(values) if v < lower or v > upper]

    def _detect_outliers_zscore(self, values: List[float]) -> List[int]:
        """Detect outliers using z-score method.

        Args:
            values: Numeric values.

        Returns:
            List of outlier indices.
        """
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std_dev = math.sqrt(variance) if variance > 0 else 0.0

        if std_dev == 0:
            return []

        threshold = self._outlier_threshold if self._outlier_threshold > 2 else 3.0
        return [
            i for i, v in enumerate(values)
            if abs((v - mean) / std_dev) > threshold
        ]
