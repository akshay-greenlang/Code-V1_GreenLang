"""
CDP Historical Tracker -- Year-over-Year Score Progression

This module implements multi-year historical tracking of CDP Climate Change
disclosure scores.  It enables year-over-year comparison, score trend analysis,
category-level progression, response carry-forward from prior submissions, and
change logging between reporting years.

Historical data is essential for demonstrating continuous improvement in CDP
scoring and understanding which categories drive year-on-year movement.

Key capabilities:
  - Year-over-year score comparison
  - Score progression visualization data (sparklines, trend lines)
  - Per-category trend analysis across years
  - Response carry-forward from previous year submissions
  - Change log between years (additions, deletions, modifications)
  - Trend detection (improving / declining / stable)
  - Best-year and peak-score identification

Example:
    >>> tracker = HistoricalTracker(config)
    >>> tracker.record_year_score("org-1", 2024, 45.0, ScoringLevel.C, True)
    >>> tracker.record_year_score("org-1", 2025, 58.0, ScoringLevel.B_MINUS, True)
    >>> result = tracker.get_progression("org-1")
    >>> print(result.score_trend)
    'improving'
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    CDPAppConfig,
    ScoringBand,
    ScoringCategory,
    ScoringLevel,
    SCORING_LEVEL_BANDS,
    SCORING_LEVEL_THRESHOLDS,
)
from .models import (
    HistoricalTrackingResult,
    YearlyScoreRecord,
    _now,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trend Detection Thresholds
# ---------------------------------------------------------------------------

IMPROVING_THRESHOLD = 2.0    # >= 2 points average annual improvement
DECLINING_THRESHOLD = -2.0   # <= -2 points average annual change
# Between these thresholds -> "stable"


class HistoricalTracker:
    """
    CDP Historical Tracker -- tracks multi-year score progression.

    Stores yearly score records, calculates trends, compares categories
    across years, and supports response carry-forward.

    Attributes:
        config: Application configuration.
        _history: Org ID -> list of YearlyScoreRecords.
        _change_logs: Org ID -> year -> list of change entries.
        _carried_forward: Questionnaire ID -> dict of carried forward data.

    Example:
        >>> tracker = HistoricalTracker(config)
        >>> tracker.record_year_score("org-1", 2024, 52.5, ScoringLevel.B_MINUS)
    """

    def __init__(self, config: CDPAppConfig) -> None:
        """Initialize the Historical Tracker."""
        self.config = config
        self._history: Dict[str, List[YearlyScoreRecord]] = {}
        self._change_logs: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
        self._carried_forward: Dict[str, Dict[str, Any]] = {}
        logger.info("HistoricalTracker initialized")

    # ------------------------------------------------------------------
    # Record Yearly Scores
    # ------------------------------------------------------------------

    def record_year_score(
        self,
        org_id: str,
        year: int,
        overall_score: float,
        level: ScoringLevel,
        submitted: bool = False,
        category_scores: Optional[Dict[str, float]] = None,
        completion_pct: float = 0.0,
    ) -> YearlyScoreRecord:
        """
        Record a yearly CDP score for an organization.

        Args:
            org_id: Organization ID.
            year: Reporting year.
            overall_score: Overall score percentage (0-100).
            level: Scoring level (D- through A).
            submitted: Whether the questionnaire was submitted to CDP.
            category_scores: Per-category scores {SC01: 72.5, SC02: 55.0, ...}.
            completion_pct: Questionnaire completion percentage.

        Returns:
            Created YearlyScoreRecord.
        """
        band = SCORING_LEVEL_BANDS.get(level, ScoringBand.DISCLOSURE)

        record = YearlyScoreRecord(
            year=year,
            overall_score=round(max(0.0, min(100.0, overall_score)), 1),
            level=level,
            band=band,
            category_scores=category_scores or {},
            completion_pct=round(max(0.0, min(100.0, completion_pct)), 1),
            submitted=submitted,
        )

        if org_id not in self._history:
            self._history[org_id] = []

        # Replace existing record for same year if present
        self._history[org_id] = [
            r for r in self._history[org_id] if r.year != year
        ]
        self._history[org_id].append(record)

        # Keep sorted by year
        self._history[org_id].sort(key=lambda r: r.year)

        logger.info(
            "Recorded %d score for org %s: %.1f%% (%s)",
            year, org_id, overall_score, level.value,
        )
        return record

    def get_year_score(
        self,
        org_id: str,
        year: int,
    ) -> Optional[YearlyScoreRecord]:
        """Get the score record for a specific year."""
        records = self._history.get(org_id, [])
        for rec in records:
            if rec.year == year:
                return rec
        return None

    # ------------------------------------------------------------------
    # Score Progression
    # ------------------------------------------------------------------

    def get_progression(
        self,
        org_id: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> HistoricalTrackingResult:
        """
        Get the multi-year score progression for an organization.

        Computes trend direction, best year, improvement rate,
        and per-category trends.

        Args:
            org_id: Organization ID.
            start_year: Optional start year filter.
            end_year: Optional end year filter.

        Returns:
            HistoricalTrackingResult with trend analysis.
        """
        records = self._history.get(org_id, [])

        # Apply year filters
        if start_year:
            records = [r for r in records if r.year >= start_year]
        if end_year:
            records = [r for r in records if r.year <= end_year]

        records.sort(key=lambda r: r.year)

        # Calculate trend
        trend = self._detect_trend(records)
        best_year, best_score = self._find_best_year(records)
        improvement_rate = self._calculate_improvement_rate(records)
        category_trends = self._build_category_trends(records)

        return HistoricalTrackingResult(
            org_id=org_id,
            years=records,
            score_trend=trend,
            best_year=best_year,
            best_score=best_score,
            improvement_rate_pct=improvement_rate,
            category_trends=category_trends,
        )

    # ------------------------------------------------------------------
    # Year-over-Year Comparison
    # ------------------------------------------------------------------

    def compare_years(
        self,
        org_id: str,
        year_a: int,
        year_b: int,
    ) -> Dict[str, Any]:
        """
        Compare scores between two years.

        Args:
            org_id: Organization ID.
            year_a: First (earlier) year.
            year_b: Second (later) year.

        Returns:
            Year-over-year comparison with deltas.
        """
        rec_a = self.get_year_score(org_id, year_a)
        rec_b = self.get_year_score(org_id, year_b)

        if not rec_a and not rec_b:
            return {
                "org_id": org_id,
                "year_a": year_a,
                "year_b": year_b,
                "available": False,
                "message": "No data for either year",
            }

        result: Dict[str, Any] = {
            "org_id": org_id,
            "year_a": year_a,
            "year_b": year_b,
            "available": True,
        }

        # Overall scores
        score_a = rec_a.overall_score if rec_a else None
        score_b = rec_b.overall_score if rec_b else None
        result["score_a"] = score_a
        result["score_b"] = score_b
        result["score_delta"] = round(score_b - score_a, 1) if score_a is not None and score_b is not None else None

        # Levels
        result["level_a"] = rec_a.level.value if rec_a else None
        result["level_b"] = rec_b.level.value if rec_b else None

        # Band movement
        result["band_a"] = rec_a.band.value if rec_a else None
        result["band_b"] = rec_b.band.value if rec_b else None

        # Completion
        result["completion_a"] = rec_a.completion_pct if rec_a else None
        result["completion_b"] = rec_b.completion_pct if rec_b else None

        # Category-level comparison
        cat_deltas: Dict[str, Dict[str, Any]] = {}
        all_categories: set = set()
        if rec_a:
            all_categories.update(rec_a.category_scores.keys())
        if rec_b:
            all_categories.update(rec_b.category_scores.keys())

        for cat in sorted(all_categories):
            cat_a = rec_a.category_scores.get(cat) if rec_a else None
            cat_b = rec_b.category_scores.get(cat) if rec_b else None
            delta = None
            if cat_a is not None and cat_b is not None:
                delta = round(cat_b - cat_a, 1)

            cat_deltas[cat] = {
                "score_a": cat_a,
                "score_b": cat_b,
                "delta": delta,
                "direction": self._direction_label(delta) if delta is not None else "unknown",
            }

        result["category_comparison"] = cat_deltas

        # Identify biggest improvements and declines
        if cat_deltas:
            sorted_by_delta = sorted(
                [(k, v) for k, v in cat_deltas.items() if v["delta"] is not None],
                key=lambda x: x[1]["delta"],
                reverse=True,
            )
            result["top_improvements"] = [
                {"category": k, "delta": v["delta"]}
                for k, v in sorted_by_delta[:3]
                if v["delta"] is not None and v["delta"] > 0
            ]
            result["top_declines"] = [
                {"category": k, "delta": v["delta"]}
                for k, v in sorted_by_delta[-3:]
                if v["delta"] is not None and v["delta"] < 0
            ]

        return result

    # ------------------------------------------------------------------
    # Score Progression Visualization Data
    # ------------------------------------------------------------------

    def get_sparkline_data(
        self,
        org_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get simplified score data for sparkline visualization.

        Returns:
            List of {year, score, level} for chart rendering.
        """
        records = self._history.get(org_id, [])
        return [
            {
                "year": r.year,
                "score": r.overall_score,
                "level": r.level.value,
            }
            for r in sorted(records, key=lambda r: r.year)
        ]

    def get_trend_line_data(
        self,
        org_id: str,
    ) -> Dict[str, Any]:
        """
        Get data for trend line visualization with linear regression.

        Returns:
            Trend line data with slope, intercept, and projected next year.
        """
        records = self._history.get(org_id, [])
        records = sorted(records, key=lambda r: r.year)

        if len(records) < 2:
            return {
                "org_id": org_id,
                "available": False,
                "message": "At least 2 years of data required for trend line",
            }

        years = [r.year for r in records]
        scores = [r.overall_score for r in records]

        # Simple linear regression
        slope, intercept = self._linear_regression(years, scores)

        # Project next year
        next_year = max(years) + 1
        projected_score = max(0.0, min(100.0, slope * next_year + intercept))

        return {
            "org_id": org_id,
            "available": True,
            "data_points": [
                {"year": y, "actual_score": s}
                for y, s in zip(years, scores)
            ],
            "trend_line": {
                "slope": round(slope, 3),
                "intercept": round(intercept, 3),
                "direction": "improving" if slope > 0 else ("declining" if slope < 0 else "flat"),
            },
            "projection": {
                "year": next_year,
                "projected_score": round(projected_score, 1),
            },
            "r_squared": round(self._r_squared(years, scores, slope, intercept), 3),
        }

    # ------------------------------------------------------------------
    # Category Trend Analysis
    # ------------------------------------------------------------------

    def get_category_trends(
        self,
        org_id: str,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get per-category trend analysis across all years.

        Args:
            org_id: Organization ID.
            categories: Specific categories to analyze; all if None.

        Returns:
            Category-level trends with direction and average change.
        """
        records = self._history.get(org_id, [])
        records = sorted(records, key=lambda r: r.year)

        if len(records) < 2:
            return {
                "org_id": org_id,
                "available": False,
                "message": "At least 2 years of data required for category trends",
            }

        # Gather all category IDs
        all_cats: set = set()
        for rec in records:
            all_cats.update(rec.category_scores.keys())

        if categories:
            all_cats = all_cats.intersection(set(categories))

        trends: Dict[str, Dict[str, Any]] = {}

        for cat in sorted(all_cats):
            cat_data = []
            for rec in records:
                if cat in rec.category_scores:
                    cat_data.append({
                        "year": rec.year,
                        "score": rec.category_scores[cat],
                    })

            if len(cat_data) < 2:
                trends[cat] = {
                    "data_points": cat_data,
                    "trend": "insufficient_data",
                    "avg_annual_change": 0.0,
                }
                continue

            # Calculate average annual change
            changes = []
            for i in range(1, len(cat_data)):
                year_gap = cat_data[i]["year"] - cat_data[i - 1]["year"]
                if year_gap > 0:
                    annual_change = (cat_data[i]["score"] - cat_data[i - 1]["score"]) / year_gap
                    changes.append(annual_change)

            avg_change = sum(changes) / len(changes) if changes else 0.0

            if avg_change >= IMPROVING_THRESHOLD:
                trend_dir = "improving"
            elif avg_change <= DECLINING_THRESHOLD:
                trend_dir = "declining"
            else:
                trend_dir = "stable"

            trends[cat] = {
                "data_points": cat_data,
                "trend": trend_dir,
                "avg_annual_change": round(avg_change, 2),
                "latest_score": cat_data[-1]["score"],
                "earliest_score": cat_data[0]["score"],
                "total_change": round(cat_data[-1]["score"] - cat_data[0]["score"], 1),
            }

        return {
            "org_id": org_id,
            "available": True,
            "categories": trends,
            "improving_count": sum(1 for t in trends.values() if t.get("trend") == "improving"),
            "declining_count": sum(1 for t in trends.values() if t.get("trend") == "declining"),
            "stable_count": sum(1 for t in trends.values() if t.get("trend") == "stable"),
        }

    # ------------------------------------------------------------------
    # Response Carry-Forward
    # ------------------------------------------------------------------

    def carry_forward_responses(
        self,
        org_id: str,
        source_year: int,
        target_questionnaire_id: str,
        response_data: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Carry forward responses from a previous year to a new questionnaire.

        Copies response content from the source year for questions that
        exist in both years.  Changes are logged for audit.

        Args:
            org_id: Organization ID.
            source_year: Year to copy responses from.
            target_questionnaire_id: Target questionnaire to populate.
            response_data: Question ID -> response content from source year.

        Returns:
            Carry-forward result with counts and change log.
        """
        carried = 0
        skipped = 0
        changes: List[Dict[str, Any]] = []

        for question_id, data in response_data.items():
            carried += 1
            changes.append({
                "question_id": question_id,
                "action": "carried_forward",
                "source_year": source_year,
                "content_preview": str(data.get("content", ""))[:100],
            })

        self._carried_forward[target_questionnaire_id] = {
            "source_year": source_year,
            "carried_count": carried,
            "response_data": response_data,
            "carried_at": _now().isoformat(),
        }

        # Log the carry-forward event
        self._log_change(
            org_id, source_year + 1,
            "carry_forward",
            f"Carried forward {carried} responses from {source_year}",
        )

        logger.info(
            "Carried forward %d responses from %d to questionnaire %s for org %s",
            carried, source_year, target_questionnaire_id, org_id,
        )

        return {
            "org_id": org_id,
            "source_year": source_year,
            "target_questionnaire_id": target_questionnaire_id,
            "carried_forward": carried,
            "skipped": skipped,
            "changes": changes,
        }

    def get_carried_forward_data(
        self,
        questionnaire_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get the carried-forward data for a questionnaire."""
        return self._carried_forward.get(questionnaire_id)

    # ------------------------------------------------------------------
    # Change Log
    # ------------------------------------------------------------------

    def log_response_change(
        self,
        org_id: str,
        year: int,
        question_id: str,
        change_type: str,
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
        changed_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Log a response change for audit trail.

        Args:
            org_id: Organization ID.
            year: Reporting year.
            question_id: Question that changed.
            change_type: Type of change (added, modified, deleted).
            old_value: Previous value.
            new_value: New value.
            changed_by: User who made the change.

        Returns:
            Change log entry.
        """
        entry = {
            "question_id": question_id,
            "change_type": change_type,
            "old_value": old_value,
            "new_value": new_value,
            "changed_by": changed_by,
            "changed_at": _now().isoformat(),
        }

        self._log_change(org_id, year, change_type, f"Q {question_id}: {change_type}", entry)
        return entry

    def get_change_log(
        self,
        org_id: str,
        year: int,
    ) -> List[Dict[str, Any]]:
        """Get the change log for a specific year."""
        org_logs = self._change_logs.get(org_id, {})
        return org_logs.get(year, [])

    def get_change_summary(
        self,
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Get a summary of changes for a year.

        Returns:
            Counts of additions, modifications, deletions, and carry-forwards.
        """
        changes = self.get_change_log(org_id, year)

        added = sum(1 for c in changes if c.get("change_type") == "added")
        modified = sum(1 for c in changes if c.get("change_type") == "modified")
        deleted = sum(1 for c in changes if c.get("change_type") == "deleted")
        carry_forward = sum(1 for c in changes if c.get("change_type") == "carry_forward")

        return {
            "org_id": org_id,
            "year": year,
            "total_changes": len(changes),
            "added": added,
            "modified": modified,
            "deleted": deleted,
            "carry_forward": carry_forward,
        }

    # ------------------------------------------------------------------
    # Multi-Year Overview
    # ------------------------------------------------------------------

    def get_multi_year_overview(
        self,
        org_id: str,
    ) -> Dict[str, Any]:
        """
        Get a comprehensive multi-year overview for dashboard display.

        Returns:
            Overview with yearly scores, progression, trends, and statistics.
        """
        progression = self.get_progression(org_id)
        sparkline = self.get_sparkline_data(org_id)
        trend_line = self.get_trend_line_data(org_id)
        category_trends = self.get_category_trends(org_id)

        records = self._history.get(org_id, [])
        years_reported = len(records)
        years_submitted = sum(1 for r in records if r.submitted)

        # Consecutive submission streak
        streak = 0
        for rec in reversed(sorted(records, key=lambda r: r.year)):
            if rec.submitted:
                streak += 1
            else:
                break

        # Level progression (list of levels in order)
        level_progression = [
            {"year": r.year, "level": r.level.value, "band": r.band.value}
            for r in sorted(records, key=lambda r: r.year)
        ]

        return {
            "org_id": org_id,
            "years_reported": years_reported,
            "years_submitted": years_submitted,
            "submission_streak": streak,
            "sparkline": sparkline,
            "trend_line": trend_line,
            "level_progression": level_progression,
            "score_trend": progression.score_trend,
            "improvement_rate_pct": progression.improvement_rate_pct,
            "best_year": progression.best_year,
            "best_score": progression.best_score,
            "category_trends_summary": {
                "improving": category_trends.get("improving_count", 0),
                "declining": category_trends.get("declining_count", 0),
                "stable": category_trends.get("stable_count", 0),
            },
        }

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _detect_trend(self, records: List[YearlyScoreRecord]) -> str:
        """Detect score trend direction from records."""
        if len(records) < 2:
            return "stable"

        # Use last 3 years if available
        recent = records[-3:] if len(records) >= 3 else records

        changes = []
        for i in range(1, len(recent)):
            year_gap = recent[i].year - recent[i - 1].year
            if year_gap > 0:
                annual_change = (recent[i].overall_score - recent[i - 1].overall_score) / year_gap
                changes.append(annual_change)

        if not changes:
            return "stable"

        avg_change = sum(changes) / len(changes)

        if avg_change >= IMPROVING_THRESHOLD:
            return "improving"
        elif avg_change <= DECLINING_THRESHOLD:
            return "declining"
        return "stable"

    def _find_best_year(
        self,
        records: List[YearlyScoreRecord],
    ) -> Tuple[Optional[int], Optional[float]]:
        """Find the year with the highest score."""
        if not records:
            return None, None

        best = max(records, key=lambda r: r.overall_score)
        return best.year, best.overall_score

    def _calculate_improvement_rate(
        self,
        records: List[YearlyScoreRecord],
    ) -> float:
        """Calculate average annual improvement rate."""
        if len(records) < 2:
            return 0.0

        first = records[0]
        last = records[-1]
        year_span = last.year - first.year

        if year_span <= 0:
            return 0.0

        total_change = last.overall_score - first.overall_score
        return round(total_change / year_span, 2)

    def _build_category_trends(
        self,
        records: List[YearlyScoreRecord],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Build per-category trend data."""
        all_cats: set = set()
        for rec in records:
            all_cats.update(rec.category_scores.keys())

        trends: Dict[str, List[Dict[str, Any]]] = {}
        for cat in sorted(all_cats):
            data_points = []
            for rec in records:
                if cat in rec.category_scores:
                    data_points.append({
                        "year": rec.year,
                        "score": rec.category_scores[cat],
                    })
            if data_points:
                trends[cat] = data_points

        return trends

    def _log_change(
        self,
        org_id: str,
        year: int,
        change_type: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an entry to the change log."""
        if org_id not in self._change_logs:
            self._change_logs[org_id] = {}
        if year not in self._change_logs[org_id]:
            self._change_logs[org_id][year] = []

        self._change_logs[org_id][year].append({
            "change_type": change_type,
            "description": description,
            "details": details or {},
            "timestamp": _now().isoformat(),
        })

    def _direction_label(self, delta: float) -> str:
        """Return direction label for a score delta."""
        if delta > 0:
            return "improved"
        elif delta < 0:
            return "declined"
        return "unchanged"

    def _linear_regression(
        self,
        x: List[int],
        y: List[float],
    ) -> Tuple[float, float]:
        """
        Simple linear regression returning (slope, intercept).

        Uses least-squares method: y = slope * x + intercept.
        """
        n = len(x)
        if n < 2:
            return 0.0, 0.0

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0, sum_y / n

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        return slope, intercept

    def _r_squared(
        self,
        x: List[int],
        y: List[float],
        slope: float,
        intercept: float,
    ) -> float:
        """Calculate R-squared (coefficient of determination)."""
        n = len(y)
        if n < 2:
            return 0.0

        mean_y = sum(y) / n
        ss_total = sum((yi - mean_y) ** 2 for yi in y)
        ss_residual = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))

        if ss_total == 0:
            return 1.0

        return max(0.0, 1.0 - ss_residual / ss_total)
