# -*- coding: utf-8 -*-
"""
Contextual Outlier Detection Engine - AGENT-DATA-013

Group-based (contextual) outlier detection that identifies values which
are outliers within their context group but may be normal globally.
Supports grouping by facility, region, sector, peer group, or custom
columns, with IQR/z-score detection within each group.

Zero-Hallucination: All calculations use deterministic Python
arithmetic. No LLM calls for numeric computations.

Example:
    >>> from greenlang.outlier_detector.contextual_detector import ContextualDetectorEngine
    >>> engine = ContextualDetectorEngine()
    >>> results = engine.detect_by_group(
    ...     records=[{"region": "EU", "emissions": 100}, {"region": "EU", "emissions": 5000}],
    ...     value_column="emissions",
    ...     group_column="region",
    ... )

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
Status: Production Ready
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from greenlang.outlier_detector.config import get_config
from greenlang.outlier_detector.models import (
    ContextType,
    ContextualResult,
    DetectionMethod,
    OutlierScore,
    SeverityLevel,
)
from greenlang.outlier_detector.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


def _safe_mean(values: List[float]) -> float:
    """Compute arithmetic mean, returning 0.0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_std(values: List[float], mean: Optional[float] = None) -> float:
    """Compute population standard deviation."""
    if len(values) < 2:
        return 0.0
    m = mean if mean is not None else _safe_mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _safe_median(values: List[float]) -> float:
    """Compute median of values."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


def _percentile(values: List[float], pct: float) -> float:
    """Compute percentile using linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0]
    k = pct * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def _severity_from_score(score: float) -> SeverityLevel:
    """Map a normalised outlier score to severity level."""
    if score >= 0.95:
        return SeverityLevel.CRITICAL
    if score >= 0.80:
        return SeverityLevel.HIGH
    if score >= 0.60:
        return SeverityLevel.MEDIUM
    if score >= 0.40:
        return SeverityLevel.LOW
    return SeverityLevel.INFO


class ContextualDetectorEngine:
    """Context-aware group-based outlier detection engine.

    Detects outliers that are anomalous within their context group
    (e.g., a facility's emissions are extreme compared to other
    facilities in the same region, even if globally they are normal).

    Attributes:
        _config: Outlier detector configuration.
        _provenance: SHA-256 provenance tracker.

    Example:
        >>> engine = ContextualDetectorEngine()
        >>> results = engine.detect_by_group(
        ...     records=[{"site": "A", "value": 10}, {"site": "A", "value": 500}],
        ...     value_column="value",
        ...     group_column="site",
        ... )
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ContextualDetectorEngine.

        Args:
            config: Optional OutlierDetectorConfig override.
        """
        self._config = config or get_config()
        self._provenance = ProvenanceTracker()
        logger.info("ContextualDetectorEngine initialized")

    # ------------------------------------------------------------------
    # Group-based detection
    # ------------------------------------------------------------------

    def detect_by_group(
        self,
        records: List[Dict[str, Any]],
        value_column: str,
        group_column: str,
        method: str = "iqr",
        context_type: ContextType = ContextType.CUSTOM,
    ) -> List[ContextualResult]:
        """Detect outliers within each group defined by group_column.

        Groups records by the group_column value, then applies the
        specified detection method within each group independently.

        Args:
            records: List of record dictionaries.
            value_column: Column containing numeric values to test.
            group_column: Column defining context groups.
            method: Detection method within groups (iqr or zscore).
            context_type: Type of context for metadata.

        Returns:
            List of ContextualResult per group.
        """
        start = time.time()
        groups = self.identify_context_groups(records, [group_column])
        results: List[ContextualResult] = []

        for group_key, indices in groups.items():
            group_records = [records[i] for i in indices]
            values = self._extract_numeric(group_records, value_column)

            if not values:
                continue

            group_stats = self._compute_single_group_stats(values)
            scores = self._score_within_context(
                values, group_stats, indices, method, value_column,
            )

            outliers_found = sum(1 for s in scores if s.is_outlier)
            provenance_hash = self._provenance.build_hash({
                "method": "contextual_" + method,
                "group": group_key,
                "column": value_column,
                "group_size": len(values),
                "outliers": outliers_found,
            })

            results.append(ContextualResult(
                context_type=context_type,
                group_key=str(group_key),
                column_name=value_column,
                group_size=len(values),
                group_mean=group_stats.get("mean", 0.0),
                group_std=group_stats.get("std", 0.0),
                outliers_found=outliers_found,
                scores=scores,
                confidence=min(1.0, 0.5 + len(values) / 100.0 * 0.5),
                provenance_hash=provenance_hash,
            ))

        elapsed = time.time() - start
        logger.debug(
            "Contextual detection: %d groups, %d total outliers, %.3fs",
            len(results),
            sum(r.outliers_found for r in results),
            elapsed,
        )
        return results

    # ------------------------------------------------------------------
    # Peer comparison detection
    # ------------------------------------------------------------------

    def detect_peer_comparison(
        self,
        records: List[Dict[str, Any]],
        value_column: str,
        peer_columns: List[str],
    ) -> List[ContextualResult]:
        """Detect outliers by comparing against peer groups.

        Groups records using multiple peer columns (e.g., sector + region)
        and detects outliers within each peer group.

        Args:
            records: List of record dictionaries.
            value_column: Column containing numeric values.
            peer_columns: Columns defining peer groups.

        Returns:
            List of ContextualResult per peer group.
        """
        start = time.time()
        groups = self.identify_context_groups(records, peer_columns)
        results: List[ContextualResult] = []

        for group_key, indices in groups.items():
            group_records = [records[i] for i in indices]
            values = self._extract_numeric(group_records, value_column)

            if len(values) < 3:
                continue

            group_stats = self._compute_single_group_stats(values)
            scores = self._score_within_context(
                values, group_stats, indices, "zscore", value_column,
            )

            outliers_found = sum(1 for s in scores if s.is_outlier)
            provenance_hash = self._provenance.build_hash({
                "method": "peer_comparison",
                "group": group_key,
                "peer_columns": peer_columns,
                "column": value_column,
                "group_size": len(values),
            })

            results.append(ContextualResult(
                context_type=ContextType.PEER_GROUP,
                group_key=str(group_key),
                column_name=value_column,
                group_size=len(values),
                group_mean=group_stats.get("mean", 0.0),
                group_std=group_stats.get("std", 0.0),
                outliers_found=outliers_found,
                scores=scores,
                confidence=min(1.0, 0.4 + len(values) / 50.0 * 0.6),
                provenance_hash=provenance_hash,
            ))

        elapsed = time.time() - start
        logger.debug("Peer comparison: %d groups, %.3fs", len(results), elapsed)
        return results

    # ------------------------------------------------------------------
    # Conditional detection
    # ------------------------------------------------------------------

    def detect_conditional(
        self,
        records: List[Dict[str, Any]],
        value_column: str,
        condition_columns: List[str],
    ) -> List[ContextualResult]:
        """Detect outliers conditional on other column values.

        Groups records by the condition columns and checks for outliers
        within each conditional group, using IQR detection.

        Args:
            records: List of record dictionaries.
            value_column: Column containing numeric values.
            condition_columns: Columns defining conditions.

        Returns:
            List of ContextualResult per condition group.
        """
        start = time.time()
        groups = self.identify_context_groups(records, condition_columns)
        results: List[ContextualResult] = []

        for group_key, indices in groups.items():
            group_records = [records[i] for i in indices]
            values = self._extract_numeric(group_records, value_column)

            if len(values) < 4:
                continue

            group_stats = self._compute_single_group_stats(values)
            scores = self._score_within_context(
                values, group_stats, indices, "iqr", value_column,
            )

            outliers_found = sum(1 for s in scores if s.is_outlier)
            provenance_hash = self._provenance.build_hash({
                "method": "conditional",
                "group": group_key,
                "condition_columns": condition_columns,
                "column": value_column,
            })

            results.append(ContextualResult(
                context_type=ContextType.CUSTOM,
                group_key=str(group_key),
                column_name=value_column,
                group_size=len(values),
                group_mean=group_stats.get("mean", 0.0),
                group_std=group_stats.get("std", 0.0),
                outliers_found=outliers_found,
                scores=scores,
                confidence=min(1.0, 0.4 + len(values) / 50.0 * 0.6),
                provenance_hash=provenance_hash,
            ))

        elapsed = time.time() - start
        logger.debug("Conditional detection: %d groups, %.3fs", len(results), elapsed)
        return results

    # ------------------------------------------------------------------
    # Group statistics
    # ------------------------------------------------------------------

    def compute_group_statistics(
        self,
        records: List[Dict[str, Any]],
        value_column: str,
        group_column: str,
    ) -> Dict[str, Dict[str, float]]:
        """Compute descriptive statistics for each group.

        Args:
            records: List of record dictionaries.
            value_column: Column containing numeric values.
            group_column: Column defining groups.

        Returns:
            Dict mapping group key to statistics dict with keys:
            count, mean, median, std, min, max, q1, q3.
        """
        groups = self.identify_context_groups(records, [group_column])
        result: Dict[str, Dict[str, float]] = {}

        for group_key, indices in groups.items():
            group_records = [records[i] for i in indices]
            values = self._extract_numeric(group_records, value_column)

            if not values:
                continue

            result[str(group_key)] = self._compute_single_group_stats(values)

        return result

    # ------------------------------------------------------------------
    # Group identification
    # ------------------------------------------------------------------

    def identify_context_groups(
        self,
        records: List[Dict[str, Any]],
        group_columns: List[str],
    ) -> Dict[str, List[int]]:
        """Identify context groups from records.

        Groups records by the unique combination of values in
        the specified group columns.

        Args:
            records: List of record dictionaries.
            group_columns: Columns to group by.

        Returns:
            Dict mapping group key (string) to list of record indices.
        """
        groups: Dict[str, List[int]] = defaultdict(list)

        for i, record in enumerate(records):
            key_parts = []
            for col in group_columns:
                val = record.get(col, "")
                key_parts.append(str(val) if val is not None else "")
            key = "|".join(key_parts)
            groups[key].append(i)

        return dict(groups)

    # ------------------------------------------------------------------
    # Within-context scoring
    # ------------------------------------------------------------------

    def _score_within_context(
        self,
        values: List[float],
        context_stats: Dict[str, float],
        original_indices: List[int],
        method: str,
        column_name: str,
    ) -> List[OutlierScore]:
        """Score values within a context group.

        Args:
            values: Numeric values in the group.
            context_stats: Pre-computed group statistics.
            original_indices: Original record indices.
            method: Detection method (iqr or zscore).
            column_name: Column name.

        Returns:
            List of OutlierScore for the group.
        """
        scores: List[OutlierScore] = []

        if method == "iqr" and len(values) >= 4:
            q1 = context_stats.get("q1", 0.0)
            q3 = context_stats.get("q3", 0.0)
            iqr = q3 - q1
            k = self._config.iqr_multiplier
            lower = q1 - k * iqr
            upper = q3 + k * iqr

            for j, v in enumerate(values):
                idx = original_indices[j]
                is_outlier = v < lower or v > upper
                if iqr > 0:
                    raw = max(0.0, max(lower - v, v - upper)) / (k * iqr)
                    score = min(1.0, raw / 2.0)
                else:
                    score = 0.0

                provenance_hash = self._provenance.build_hash({
                    "method": "contextual_iqr", "index": idx, "value": v,
                })

                scores.append(OutlierScore(
                    record_index=idx,
                    column_name=column_name,
                    value=v,
                    method=DetectionMethod.CONTEXTUAL,
                    score=score,
                    is_outlier=is_outlier,
                    threshold=k,
                    severity=_severity_from_score(score),
                    details={"context_method": "iqr", "group_q1": q1,
                             "group_q3": q3, "group_iqr": iqr},
                    confidence=min(1.0, 0.5 + score * 0.5),
                    provenance_hash=provenance_hash,
                ))
        else:
            # z-score within context
            mean = context_stats.get("mean", 0.0)
            std = context_stats.get("std", 0.0)
            t = self._config.zscore_threshold

            for j, v in enumerate(values):
                idx = original_indices[j]
                if std > 0:
                    z = abs(v - mean) / std
                    score = min(1.0, z / (t * 2.0))
                else:
                    z = 0.0
                    score = 0.0

                is_outlier = z > t
                provenance_hash = self._provenance.build_hash({
                    "method": "contextual_zscore", "index": idx, "value": v,
                })

                scores.append(OutlierScore(
                    record_index=idx,
                    column_name=column_name,
                    value=v,
                    method=DetectionMethod.CONTEXTUAL,
                    score=score,
                    is_outlier=is_outlier,
                    threshold=t,
                    severity=_severity_from_score(score),
                    details={"context_method": "zscore",
                             "group_mean": mean, "group_std": std, "z": z},
                    confidence=min(1.0, 0.5 + score * 0.5),
                    provenance_hash=provenance_hash,
                ))

        return scores

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_numeric(
        self,
        records: List[Dict[str, Any]],
        column: str,
    ) -> List[float]:
        """Extract numeric values from a column, skipping non-numeric.

        Args:
            records: Record dictionaries.
            column: Column name to extract.

        Returns:
            List of float values.
        """
        values: List[float] = []
        for rec in records:
            val = rec.get(column)
            if val is not None:
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    pass
        return values

    def _compute_single_group_stats(
        self,
        values: List[float],
    ) -> Dict[str, float]:
        """Compute statistics for a single group.

        Args:
            values: Numeric values.

        Returns:
            Dict with count, mean, median, std, min, max, q1, q3.
        """
        if not values:
            return {
                "count": 0, "mean": 0.0, "median": 0.0, "std": 0.0,
                "min": 0.0, "max": 0.0, "q1": 0.0, "q3": 0.0,
            }

        mean = _safe_mean(values)
        return {
            "count": float(len(values)),
            "mean": mean,
            "median": _safe_median(values),
            "std": _safe_std(values, mean),
            "min": float(min(values)),
            "max": float(max(values)),
            "q1": _percentile(values, 0.25),
            "q3": _percentile(values, 0.75),
        }


__all__ = [
    "ContextualDetectorEngine",
]
