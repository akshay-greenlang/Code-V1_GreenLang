# -*- coding: utf-8 -*-
"""
Timeliness Tracker Engine - AGENT-DATA-010: Data Quality Profiler (GL-DATA-X-013)

Data freshness and staleness scoring with SLA compliance checking.
Implements a configurable freshness degradation curve, per-record
freshness analysis, stale record detection, update frequency analysis,
and SLA compliance reporting.

Zero-Hallucination Guarantees:
    - All freshness scores use deterministic piecewise-linear degradation
    - Timestamp parsing uses standard Python datetime only
    - No ML/LLM calls in the analysis path
    - SHA-256 provenance on every analysis mutation
    - Thread-safe in-memory storage

Freshness Degradation Curve:
    - age <= excellent_hours (1h):   score = 1.0
    - age <= good_hours (6h):        linear 1.0 -> 0.85
    - age <= fair_hours (24h):       linear 0.85 -> 0.70
    - age <= poor_hours (72h):       linear 0.70 -> 0.50
    - age > poor_hours:              score = 0.0

Example:
    >>> from greenlang.data_quality_profiler.timeliness_tracker import TimelinessTracker
    >>> tracker = TimelinessTracker()
    >>> result = tracker.check_freshness("my_dataset", "2026-02-09T10:00:00Z", sla_hours=24)
    >>> print(result["freshness_score"], result["is_fresh"])

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
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "TimelinessTracker",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_id(prefix: str = "TML") -> str:
    """Generate a unique identifier with the given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        String of the form ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash for a timeliness operation.

    Args:
        operation: Name of the operation.
        data_repr: Serialised representation of the data involved.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse a value into a UTC datetime object.

    Supports ISO-8601 strings, Unix timestamps, and datetime objects.

    Args:
        value: Value to parse (str, int, float, or datetime).

    Returns:
        Parsed UTC datetime, or None if unparseable.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Try datetime.fromisoformat first (handles microseconds, tz)
        try:
            # Handle 'Z' suffix (Python < 3.11 does not parse 'Z')
            iso_s = s.replace("Z", "+00:00") if s.endswith("Z") else s
            dt = datetime.fromisoformat(iso_s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            pass
        # Fallback: try strptime with common formats
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y",
        ):
            try:
                dt = datetime.strptime(s, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        # Try Unix timestamp string
        try:
            ts = float(s)
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            pass
    return None


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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITY_CRITICAL = "critical"
SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"
SEVERITY_LOW = "low"
SEVERITY_INFO = "info"

# Default freshness degradation thresholds (hours)
_DEFAULT_EXCELLENT_HOURS = 1.0
_DEFAULT_GOOD_HOURS = 6.0
_DEFAULT_FAIR_HOURS = 24.0
_DEFAULT_POOR_HOURS = 72.0


# ---------------------------------------------------------------------------
# TimelinessTracker Engine
# ---------------------------------------------------------------------------


class TimelinessTracker:
    """Data freshness and staleness scoring engine.

    Computes freshness scores using a configurable piecewise-linear
    degradation curve, analyses per-record freshness, detects stale
    records, computes update frequency statistics, and checks SLA
    compliance.

    Thread-safe: all mutations to internal storage are protected by
    a threading lock. SHA-256 provenance hashes on every analysis.

    Attributes:
        _config: Configuration dictionary.
        _lock: Threading lock for thread-safe storage access.
        _checks: In-memory storage of freshness checks.
        _stats: Aggregate tracking statistics.

    Example:
        >>> tracker = TimelinessTracker()
        >>> result = tracker.check_freshness("ds", "2026-02-09T12:00:00Z", 24)
        >>> assert 0.0 <= result["freshness_score"] <= 1.0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TimelinessTracker.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``excellent_hours``: float (default 1.0)
                - ``good_hours``: float (default 6.0)
                - ``fair_hours``: float (default 24.0)
                - ``poor_hours``: float (default 72.0)
                - ``default_sla_hours``: float (default 24.0)
        """
        self._config = config or {}
        self._excellent_hours: float = self._config.get(
            "excellent_hours", _DEFAULT_EXCELLENT_HOURS
        )
        self._good_hours: float = self._config.get(
            "good_hours", _DEFAULT_GOOD_HOURS
        )
        self._fair_hours: float = self._config.get(
            "fair_hours", _DEFAULT_FAIR_HOURS
        )
        self._poor_hours: float = self._config.get(
            "poor_hours", _DEFAULT_POOR_HOURS
        )
        self._default_sla: float = self._config.get("default_sla_hours", 24.0)
        self._lock = threading.Lock()
        self._checks: Dict[str, Dict[str, Any]] = {}
        self._stats: Dict[str, Any] = {
            "checks_completed": 0,
            "total_fresh": 0,
            "total_stale": 0,
            "total_check_time_ms": 0.0,
        }
        logger.info(
            "TimelinessTracker initialized: excellent=%.1fh, good=%.1fh, "
            "fair=%.1fh, poor=%.1fh, default_sla=%.1fh",
            self._excellent_hours, self._good_hours,
            self._fair_hours, self._poor_hours, self._default_sla,
        )

    # ------------------------------------------------------------------
    # Public API - Dataset Freshness Check
    # ------------------------------------------------------------------

    def check_freshness(
        self,
        dataset_name: str,
        last_updated: Any,
        sla_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Check freshness of a dataset based on its last update time.

        Args:
            dataset_name: Human-readable dataset name.
            last_updated: Timestamp of last update (str, datetime, or unix).
            sla_hours: SLA threshold in hours. Defaults to configured value.

        Returns:
            FreshnessResult dict with: check_id, dataset_name,
            freshness_score, age_hours, sla_hours, is_fresh,
            freshness_level, provenance_hash.

        Raises:
            ValueError: If last_updated cannot be parsed.
        """
        start = time.monotonic()
        check_id = _generate_id("TML")
        sla = sla_hours if sla_hours is not None else self._default_sla

        ts = _parse_timestamp(last_updated)
        if ts is None:
            raise ValueError(
                f"Cannot parse last_updated timestamp: {last_updated!r}"
            )

        now = _utcnow()
        age_delta = now - ts
        age_hours = age_delta.total_seconds() / 3600.0
        if age_hours < 0:
            age_hours = 0.0

        freshness_score = self.compute_freshness_score(age_hours, sla)
        freshness_level = self._classify_freshness(age_hours)
        is_fresh = age_hours <= sla

        provenance_data = json.dumps({
            "check_id": check_id,
            "dataset_name": dataset_name,
            "age_hours": age_hours,
            "sla_hours": sla,
            "freshness_score": freshness_score,
        }, sort_keys=True, default=str)
        provenance_hash = _compute_provenance("check_freshness", provenance_data)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        result: Dict[str, Any] = {
            "check_id": check_id,
            "dataset_name": dataset_name,
            "last_updated": ts.isoformat(),
            "checked_at": now.isoformat(),
            "age_hours": round(age_hours, 2),
            "sla_hours": sla,
            "freshness_score": round(freshness_score, 4),
            "freshness_level": freshness_level,
            "is_fresh": is_fresh,
            "sla_compliant": is_fresh,
            "provenance_hash": provenance_hash,
            "check_time_ms": round(elapsed_ms, 2),
            "created_at": now.isoformat(),
        }

        with self._lock:
            self._checks[check_id] = result
            self._stats["checks_completed"] += 1
            if is_fresh:
                self._stats["total_fresh"] += 1
            else:
                self._stats["total_stale"] += 1
            self._stats["total_check_time_ms"] += elapsed_ms

        logger.info(
            "Freshness check: id=%s, dataset=%s, age=%.1fh, score=%.4f, fresh=%s",
            check_id, dataset_name, age_hours, freshness_score, is_fresh,
        )
        return result

    # ------------------------------------------------------------------
    # Freshness Score Computation
    # ------------------------------------------------------------------

    def compute_freshness_score(
        self,
        age_hours: float,
        sla_hours: Optional[float] = None,
    ) -> float:
        """Compute freshness score using a piecewise-linear degradation curve.

        Degradation curve:
            - age <= excellent_hours: 1.0
            - age <= good_hours: linear 1.0 -> 0.85
            - age <= fair_hours: linear 0.85 -> 0.70
            - age <= poor_hours: linear 0.70 -> 0.50
            - age > poor_hours: 0.0

        Args:
            age_hours: Data age in hours.
            sla_hours: Optional SLA threshold (not used in curve but
                available for overrides in subclasses).

        Returns:
            Float between 0.0 and 1.0.
        """
        if age_hours <= 0:
            return 1.0
        if age_hours <= self._excellent_hours:
            return 1.0
        if age_hours <= self._good_hours:
            return self._interpolate(
                age_hours, self._excellent_hours, self._good_hours, 1.0, 0.85
            )
        if age_hours <= self._fair_hours:
            return self._interpolate(
                age_hours, self._good_hours, self._fair_hours, 0.85, 0.70
            )
        if age_hours <= self._poor_hours:
            return self._interpolate(
                age_hours, self._fair_hours, self._poor_hours, 0.70, 0.50
            )
        return 0.0

    def _interpolate(
        self,
        value: float,
        low_bound: float,
        high_bound: float,
        low_score: float,
        high_score: float,
    ) -> float:
        """Linear interpolation between two score boundaries.

        Args:
            value: Current value (age_hours).
            low_bound: Lower boundary of the interval.
            high_bound: Upper boundary of the interval.
            low_score: Score at the lower boundary.
            high_score: Score at the upper boundary.

        Returns:
            Interpolated score.
        """
        if high_bound == low_bound:
            return low_score
        fraction = (value - low_bound) / (high_bound - low_bound)
        return low_score + fraction * (high_score - low_score)

    def _classify_freshness(self, age_hours: float) -> str:
        """Classify freshness level based on age.

        Args:
            age_hours: Data age in hours.

        Returns:
            Level string: excellent, good, fair, poor, stale.
        """
        if age_hours <= self._excellent_hours:
            return "excellent"
        if age_hours <= self._good_hours:
            return "good"
        if age_hours <= self._fair_hours:
            return "fair"
        if age_hours <= self._poor_hours:
            return "poor"
        return "stale"

    # ------------------------------------------------------------------
    # Field-Level Freshness
    # ------------------------------------------------------------------

    def check_field_freshness(
        self,
        data: List[Dict[str, Any]],
        timestamp_column: str,
    ) -> Dict[str, Any]:
        """Analyse per-record freshness based on a timestamp column.

        Args:
            data: List of row dictionaries.
            timestamp_column: Column name containing timestamps.

        Returns:
            Dict with: per-record freshness scores, summary statistics,
            stale count, freshness distribution.

        Raises:
            ValueError: If data is empty or timestamp_column not found.
        """
        if not data:
            raise ValueError("Cannot analyse empty dataset")

        now = _utcnow()
        record_scores: List[Dict[str, Any]] = []
        ages: List[float] = []
        parse_failures = 0

        for idx, row in enumerate(data):
            ts_val = row.get(timestamp_column)
            ts = _parse_timestamp(ts_val)

            if ts is None:
                parse_failures += 1
                record_scores.append({
                    "row_index": idx,
                    "timestamp": None,
                    "age_hours": None,
                    "freshness_score": 0.0,
                    "freshness_level": "unknown",
                    "parse_error": True,
                })
                continue

            age_hours = max(0.0, (now - ts).total_seconds() / 3600.0)
            score = self.compute_freshness_score(age_hours, None)
            level = self._classify_freshness(age_hours)
            ages.append(age_hours)

            record_scores.append({
                "row_index": idx,
                "timestamp": ts.isoformat(),
                "age_hours": round(age_hours, 2),
                "freshness_score": round(score, 4),
                "freshness_level": level,
                "parse_error": False,
            })

        # Summary statistics
        if ages:
            scores_only = [rs["freshness_score"] for rs in record_scores if not rs.get("parse_error")]
            summary: Dict[str, Any] = {
                "min_age_hours": round(min(ages), 2),
                "max_age_hours": round(max(ages), 2),
                "mean_age_hours": round(statistics.mean(ages), 2),
                "median_age_hours": round(statistics.median(ages), 2),
                "stddev_age_hours": round(_safe_stdev(ages), 2),
                "min_score": round(min(scores_only), 4) if scores_only else 0.0,
                "max_score": round(max(scores_only), 4) if scores_only else 0.0,
                "mean_score": round(statistics.mean(scores_only), 4) if scores_only else 0.0,
            }
        else:
            summary = {
                "min_age_hours": 0.0, "max_age_hours": 0.0,
                "mean_age_hours": 0.0, "median_age_hours": 0.0,
                "stddev_age_hours": 0.0,
                "min_score": 0.0, "max_score": 0.0, "mean_score": 0.0,
            }

        # Distribution
        levels = [rs["freshness_level"] for rs in record_scores]
        level_dist: Dict[str, int] = {}
        for lv in levels:
            level_dist[lv] = level_dist.get(lv, 0) + 1

        provenance_data = json.dumps({
            "timestamp_column": timestamp_column,
            "row_count": len(data),
            "parse_failures": parse_failures,
        }, sort_keys=True, default=str)
        provenance_hash = _compute_provenance("field_freshness", provenance_data)

        return {
            "timestamp_column": timestamp_column,
            "total_records": len(data),
            "parsed_records": len(ages),
            "parse_failures": parse_failures,
            "record_scores": record_scores,
            "summary": summary,
            "freshness_distribution": level_dist,
            "provenance_hash": provenance_hash,
        }

    # ------------------------------------------------------------------
    # Timeliness Score
    # ------------------------------------------------------------------

    def compute_timeliness_score(
        self,
        data: List[Dict[str, Any]],
        timestamp_columns: List[str],
    ) -> float:
        """Compute overall timeliness score across multiple timestamp columns.

        Averages the mean freshness scores from each timestamp column.

        Args:
            data: List of row dictionaries.
            timestamp_columns: List of column names containing timestamps.

        Returns:
            Float between 0.0 and 1.0.
        """
        if not data or not timestamp_columns:
            return 1.0

        column_scores: List[float] = []
        now = _utcnow()

        for col in timestamp_columns:
            scores: List[float] = []
            for row in data:
                ts = _parse_timestamp(row.get(col))
                if ts is not None:
                    age = max(0.0, (now - ts).total_seconds() / 3600.0)
                    scores.append(self.compute_freshness_score(age, None))

            if scores:
                column_scores.append(statistics.mean(scores))

        if not column_scores:
            return 1.0

        return statistics.mean(column_scores)

    # ------------------------------------------------------------------
    # Stale Record Detection
    # ------------------------------------------------------------------

    def detect_stale_records(
        self,
        data: List[Dict[str, Any]],
        timestamp_column: str,
        threshold_hours: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Detect records whose timestamp exceeds the staleness threshold.

        Args:
            data: List of row dictionaries.
            timestamp_column: Column name containing timestamps.
            threshold_hours: Age threshold in hours. Defaults to poor_hours.

        Returns:
            List of stale record dicts with: row_index, timestamp,
            age_hours, freshness_score.
        """
        threshold = threshold_hours if threshold_hours is not None else self._poor_hours
        now = _utcnow()
        stale: List[Dict[str, Any]] = []

        for idx, row in enumerate(data):
            ts = _parse_timestamp(row.get(timestamp_column))
            if ts is None:
                stale.append({
                    "row_index": idx,
                    "timestamp": None,
                    "age_hours": None,
                    "freshness_score": 0.0,
                    "reason": "unparseable_timestamp",
                })
                continue

            age = max(0.0, (now - ts).total_seconds() / 3600.0)
            if age > threshold:
                stale.append({
                    "row_index": idx,
                    "timestamp": ts.isoformat(),
                    "age_hours": round(age, 2),
                    "freshness_score": round(
                        self.compute_freshness_score(age, None), 4
                    ),
                    "reason": f"exceeds_{threshold}h_threshold",
                })

        return stale

    # ------------------------------------------------------------------
    # Update Frequency Analysis
    # ------------------------------------------------------------------

    def compute_update_frequency(
        self,
        timestamps: List[Any],
    ) -> Dict[str, Any]:
        """Compute update frequency statistics from a list of timestamps.

        Calculates intervals between consecutive timestamps and returns
        frequency statistics.

        Args:
            timestamps: List of timestamp values (will be parsed and sorted).

        Returns:
            Dict with: total_updates, mean_interval_hours,
            median_interval_hours, stddev_interval_hours,
            min_interval_hours, max_interval_hours, regularity_score.
        """
        parsed: List[datetime] = []
        for ts in timestamps:
            dt = _parse_timestamp(ts)
            if dt is not None:
                parsed.append(dt)

        if len(parsed) < 2:
            return {
                "total_updates": len(parsed),
                "mean_interval_hours": 0.0,
                "median_interval_hours": 0.0,
                "stddev_interval_hours": 0.0,
                "min_interval_hours": 0.0,
                "max_interval_hours": 0.0,
                "regularity_score": 1.0,
            }

        # Sort and compute intervals
        parsed.sort()
        intervals: List[float] = []
        for i in range(1, len(parsed)):
            delta = (parsed[i] - parsed[i - 1]).total_seconds() / 3600.0
            intervals.append(max(0.0, delta))

        mean_interval = statistics.mean(intervals)
        median_interval = statistics.median(intervals)
        std_interval = _safe_stdev(intervals)

        # Regularity score: 1.0 means perfectly regular intervals
        # Uses coefficient of variation (lower = more regular)
        if mean_interval > 0:
            cv = std_interval / mean_interval
            regularity = max(0.0, 1.0 - min(cv, 1.0))
        else:
            regularity = 1.0

        return {
            "total_updates": len(parsed),
            "total_intervals": len(intervals),
            "mean_interval_hours": round(mean_interval, 2),
            "median_interval_hours": round(median_interval, 2),
            "stddev_interval_hours": round(std_interval, 2),
            "min_interval_hours": round(min(intervals), 2),
            "max_interval_hours": round(max(intervals), 2),
            "regularity_score": round(regularity, 4),
        }

    # ------------------------------------------------------------------
    # SLA Compliance
    # ------------------------------------------------------------------

    def check_sla_compliance(
        self,
        freshness_results: List[Dict[str, Any]],
        sla_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Check SLA compliance across multiple freshness check results.

        Args:
            freshness_results: List of freshness result dicts from
                check_freshness calls.
            sla_hours: SLA threshold in hours. Overrides per-result values.

        Returns:
            Compliance dict with: total_datasets, compliant_count,
            non_compliant_count, compliance_rate, non_compliant_datasets.
        """
        sla = sla_hours if sla_hours is not None else self._default_sla
        compliant = 0
        non_compliant_list: List[Dict[str, Any]] = []

        for fr in freshness_results:
            age = fr.get("age_hours", 0.0)
            ds_name = fr.get("dataset_name", "unknown")

            if age is not None and age <= sla:
                compliant += 1
            else:
                non_compliant_list.append({
                    "dataset_name": ds_name,
                    "age_hours": age,
                    "sla_hours": sla,
                    "overage_hours": round((age or 0.0) - sla, 2),
                })

        total = len(freshness_results)
        non_compliant = total - compliant
        compliance_rate = compliant / total if total > 0 else 1.0

        provenance_data = json.dumps({
            "total": total,
            "compliant": compliant,
            "sla_hours": sla,
        }, sort_keys=True, default=str)
        provenance_hash = _compute_provenance("sla_compliance", provenance_data)

        return {
            "total_datasets": total,
            "compliant_count": compliant,
            "non_compliant_count": non_compliant,
            "compliance_rate": round(compliance_rate, 4),
            "sla_hours": sla,
            "non_compliant_datasets": non_compliant_list,
            "provenance_hash": provenance_hash,
        }

    # ------------------------------------------------------------------
    # Issue Generation
    # ------------------------------------------------------------------

    def generate_timeliness_issues(
        self,
        data: List[Dict[str, Any]],
        timestamp_columns: List[str],
    ) -> List[Dict[str, Any]]:
        """Generate timeliness quality issues for a dataset.

        Args:
            data: List of row dictionaries.
            timestamp_columns: List of timestamp column names.

        Returns:
            List of issue dicts with: issue_id, type, severity,
            column, message, details.
        """
        if not data or not timestamp_columns:
            return []

        issues: List[Dict[str, Any]] = []
        now = _utcnow()

        for col in timestamp_columns:
            ages: List[float] = []
            parse_failures = 0

            for row in data:
                ts = _parse_timestamp(row.get(col))
                if ts is None:
                    parse_failures += 1
                else:
                    ages.append(max(0.0, (now - ts).total_seconds() / 3600.0))

            if parse_failures > 0:
                rate = parse_failures / len(data)
                severity = SEVERITY_HIGH if rate > 0.1 else SEVERITY_MEDIUM
                issues.append({
                    "issue_id": _generate_id("ISS"),
                    "type": "unparseable_timestamps",
                    "severity": severity,
                    "column": col,
                    "message": (
                        f"Column '{col}' has {parse_failures}/{len(data)} "
                        f"({rate:.1%}) unparseable timestamps"
                    ),
                    "details": {
                        "parse_failures": parse_failures,
                        "total_records": len(data),
                        "failure_rate": round(rate, 4),
                    },
                    "created_at": now.isoformat(),
                })

            if ages:
                max_age = max(ages)
                mean_age = statistics.mean(ages)
                stale_count = sum(1 for a in ages if a > self._poor_hours)

                if stale_count > 0:
                    stale_rate = stale_count / len(ages)
                    severity = SEVERITY_CRITICAL if stale_rate > 0.5 else (
                        SEVERITY_HIGH if stale_rate > 0.1 else SEVERITY_MEDIUM
                    )
                    issues.append({
                        "issue_id": _generate_id("ISS"),
                        "type": "stale_data",
                        "severity": severity,
                        "column": col,
                        "message": (
                            f"Column '{col}' has {stale_count}/{len(ages)} "
                            f"({stale_rate:.1%}) stale records "
                            f"(>{self._poor_hours}h old)"
                        ),
                        "details": {
                            "stale_count": stale_count,
                            "total_parseable": len(ages),
                            "stale_rate": round(stale_rate, 4),
                            "max_age_hours": round(max_age, 2),
                            "mean_age_hours": round(mean_age, 2),
                        },
                        "created_at": now.isoformat(),
                    })

                if max_age > self._fair_hours:
                    issues.append({
                        "issue_id": _generate_id("ISS"),
                        "type": "data_staleness_warning",
                        "severity": SEVERITY_LOW,
                        "column": col,
                        "message": (
                            f"Column '{col}' max age is {max_age:.1f}h "
                            f"(mean {mean_age:.1f}h)"
                        ),
                        "details": {
                            "max_age_hours": round(max_age, 2),
                            "mean_age_hours": round(mean_age, 2),
                        },
                        "created_at": now.isoformat(),
                    })

        return issues

    # ------------------------------------------------------------------
    # Storage and Retrieval
    # ------------------------------------------------------------------

    def get_check(self, check_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored freshness check by ID.

        Args:
            check_id: The check identifier.

        Returns:
            Check dict or None if not found.
        """
        with self._lock:
            return self._checks.get(check_id)

    def list_checks(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List stored checks with pagination.

        Args:
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of check dicts sorted by creation time descending.
        """
        with self._lock:
            all_checks = sorted(
                self._checks.values(),
                key=lambda c: c.get("created_at", ""),
                reverse=True,
            )
            return all_checks[offset:offset + limit]

    def delete_check(self, check_id: str) -> bool:
        """Delete a stored check.

        Args:
            check_id: The check identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if check_id in self._checks:
                del self._checks[check_id]
                logger.info("Timeliness check deleted: %s", check_id)
                return True
            return False

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate tracking statistics.

        Returns:
            Dictionary with counters and totals for all timeliness
            checks performed by this engine instance.
        """
        with self._lock:
            completed = self._stats["checks_completed"]
            avg_time = (
                self._stats["total_check_time_ms"] / completed
                if completed > 0 else 0.0
            )
            fresh_rate = (
                self._stats["total_fresh"] / completed
                if completed > 0 else 0.0
            )
            return {
                "checks_completed": completed,
                "total_fresh": self._stats["total_fresh"],
                "total_stale": self._stats["total_stale"],
                "freshness_rate": round(fresh_rate, 4),
                "total_check_time_ms": round(
                    self._stats["total_check_time_ms"], 2
                ),
                "avg_check_time_ms": round(avg_time, 2),
                "stored_checks": len(self._checks),
                "timestamp": _utcnow().isoformat(),
            }
