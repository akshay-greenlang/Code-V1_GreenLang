# -*- coding: utf-8 -*-
"""
SOC 2 Self-Assessment Scorer - SEC-009

Implements the Scorer class for calculating SOC 2 readiness scores from
self-assessments. Provides deterministic scoring algorithms for overall
maturity, category-level scores, and readiness percentages.

All scoring is ZERO-HALLUCINATION - calculations use only Python arithmetic
and defined formulas. No LLM inference is used for numeric calculations.

Scoring Scale:
    - 0 (NOT_IMPLEMENTED): Control does not exist
    - 1 (PARTIAL): Control exists but incomplete
    - 2 (IMPLEMENTED): Control implemented and documented
    - 3 (TESTED): Control tested but not audit-ready
    - 4 (COMPLIANT): Control fully audit-ready

Example:
    >>> from greenlang.infrastructure.soc2_preparation.self_assessment import Scorer
    >>> scorer = Scorer()
    >>> overall = scorer.calculate_overall_score(assessment)
    >>> print(f"SOC 2 Readiness: {overall:.1f}%")
    >>> readiness = scorer.get_readiness_percentage(assessment)
    >>> print(f"Audit Ready: {readiness:.1f}%")

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from greenlang.infrastructure.soc2_preparation.models import (
    Assessment,
    AssessmentCriteria,
    ScoreLevel,
)
from greenlang.infrastructure.soc2_preparation.self_assessment.criteria import (
    CATEGORY_WEIGHTS,
    TSC_CRITERIA,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score Level (Re-export for convenience)
# ---------------------------------------------------------------------------


class MaturityLevel(IntEnum):
    """SOC 2 control maturity levels.

    Aligned with ScoreLevel from models for consistency.
    """

    NOT_IMPLEMENTED = 0
    """Control does not exist or is completely absent."""

    PARTIAL = 1
    """Control exists but is incomplete, informal, or inconsistent."""

    IMPLEMENTED = 2
    """Control is implemented and documented but not formally tested."""

    TESTED = 3
    """Control is implemented, documented, and tested but not yet audit-ready."""

    COMPLIANT = 4
    """Control is fully implemented, tested, documented, and audit-ready."""


# ---------------------------------------------------------------------------
# Score Thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreThresholds:
    """Thresholds for score interpretation.

    Defines the boundaries for categorizing overall scores into
    readiness levels (not ready, partial, ready, audit-ready).
    """

    not_ready_max: float = 25.0
    """Maximum score to be considered "not ready" (0-25%)."""

    partial_max: float = 50.0
    """Maximum score to be considered "partial readiness" (26-50%)."""

    ready_max: float = 75.0
    """Maximum score to be considered "ready with gaps" (51-75%)."""

    audit_ready_min: float = 75.0
    """Minimum score to be considered "audit-ready" (>75%)."""

    compliant_min: float = 90.0
    """Minimum score to be considered "fully compliant" (>90%)."""


DEFAULT_THRESHOLDS = ScoreThresholds()


# ---------------------------------------------------------------------------
# Scorer Class
# ---------------------------------------------------------------------------


class Scorer:
    """SOC 2 readiness scorer.

    Calculates maturity scores from self-assessment results using
    deterministic formulas. All calculations are based on:
    - Individual criterion scores (0-4)
    - Category weights
    - Evidence coverage ratios

    This class implements ZERO-HALLUCINATION scoring - no LLM inference
    is used. All scores are derived from Python arithmetic.

    Attributes:
        thresholds: Score interpretation thresholds.
        category_weights: Weights for each TSC category.

    Example:
        >>> scorer = Scorer()
        >>> overall = scorer.calculate_overall_score(assessment)
        >>> category_scores = scorer.calculate_all_category_scores(assessment)
        >>> status = scorer.score_to_status(overall)
    """

    def __init__(
        self,
        thresholds: Optional[ScoreThresholds] = None,
        category_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize the scorer.

        Args:
            thresholds: Optional custom score thresholds.
            category_weights: Optional custom category weights.
        """
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.category_weights = category_weights or CATEGORY_WEIGHTS

    def calculate_overall_score(self, assessment: Assessment) -> float:
        """Calculate weighted overall maturity score.

        Computes a weighted average of all criterion scores, where weights
        are determined by category importance. Returns a score on a 0-100
        scale.

        Formula:
            overall = (sum(criterion_score * category_weight) /
                      sum(category_weight)) * 25

        The multiplication by 25 converts from the 0-4 scale to 0-100.

        Args:
            assessment: The assessment to score.

        Returns:
            Overall score as percentage (0-100).

        Example:
            >>> score = scorer.calculate_overall_score(assessment)
            >>> print(f"Overall Score: {score:.1f}%")
        """
        if not assessment.criteria:
            logger.warning("Assessment has no criteria to score")
            return 0.0

        total_weighted_score = Decimal("0")
        total_weight = Decimal("0")

        for criterion in assessment.criteria:
            # Get category weight
            criterion_def = TSC_CRITERIA.get(criterion.criterion_id, {})
            category = criterion_def.get("category", "security")
            weight = Decimal(str(self.category_weights.get(category, 1.0)))

            # Accumulate weighted score
            score = Decimal(str(criterion.score))
            total_weighted_score += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        # Convert from 0-4 scale to 0-100 scale
        raw_score = (total_weighted_score / total_weight) * Decimal("25")
        overall = float(min(raw_score, Decimal("100")))

        logger.debug(
            "Calculated overall score: %.2f%% from %d criteria",
            overall,
            len(assessment.criteria),
        )

        return round(overall, 2)

    def calculate_category_score(
        self,
        assessment: Assessment,
        category: str,
    ) -> float:
        """Calculate score for a specific TSC category.

        Computes the average score for all criteria within a category.
        Returns a score on a 0-100 scale.

        Args:
            assessment: The assessment to score.
            category: The category name (e.g., "security", "availability").

        Returns:
            Category score as percentage (0-100).

        Example:
            >>> security_score = scorer.calculate_category_score(
            ...     assessment, "security"
            ... )
        """
        category_criteria = [
            c for c in assessment.criteria
            if TSC_CRITERIA.get(c.criterion_id, {}).get("category") == category
        ]

        if not category_criteria:
            return 0.0

        total_score = sum(c.score for c in category_criteria)
        avg_score = total_score / len(category_criteria)

        # Convert from 0-4 to 0-100 scale
        return round(avg_score * 25, 2)

    def calculate_all_category_scores(
        self,
        assessment: Assessment,
    ) -> Dict[str, float]:
        """Calculate scores for all categories in an assessment.

        Args:
            assessment: The assessment to score.

        Returns:
            Dictionary mapping category names to scores (0-100).

        Example:
            >>> scores = scorer.calculate_all_category_scores(assessment)
            >>> for cat, score in scores.items():
            ...     print(f"{cat}: {score:.1f}%")
        """
        categories = set()
        for criterion in assessment.criteria:
            criterion_def = TSC_CRITERIA.get(criterion.criterion_id, {})
            category = criterion_def.get("category")
            if category:
                categories.add(category)

        return {
            category: self.calculate_category_score(assessment, category)
            for category in sorted(categories)
        }

    def calculate_subcategory_scores(
        self,
        assessment: Assessment,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate scores for all subcategories.

        Provides granular scoring at the subcategory level
        (e.g., "access_controls", "change_management").

        Args:
            assessment: The assessment to score.

        Returns:
            Nested dictionary: category -> subcategory -> score.
        """
        # Group criteria by category and subcategory
        subcategory_scores: Dict[str, Dict[str, List[int]]] = {}

        for criterion in assessment.criteria:
            criterion_def = TSC_CRITERIA.get(criterion.criterion_id, {})
            category = criterion_def.get("category", "unknown")
            subcategory = criterion_def.get("subcategory", "unknown")

            if category not in subcategory_scores:
                subcategory_scores[category] = {}
            if subcategory not in subcategory_scores[category]:
                subcategory_scores[category][subcategory] = []

            subcategory_scores[category][subcategory].append(criterion.score)

        # Calculate averages
        result: Dict[str, Dict[str, float]] = {}
        for category, subcats in subcategory_scores.items():
            result[category] = {}
            for subcat, scores in subcats.items():
                avg = sum(scores) / len(scores) if scores else 0
                result[category][subcat] = round(avg * 25, 2)

        return result

    def get_readiness_percentage(self, assessment: Assessment) -> float:
        """Calculate audit readiness percentage.

        Measures what percentage of criteria have achieved COMPLIANT (4)
        status, indicating they are ready for external audit.

        Args:
            assessment: The assessment to evaluate.

        Returns:
            Percentage of criteria at COMPLIANT level (0-100).

        Example:
            >>> readiness = scorer.get_readiness_percentage(assessment)
            >>> print(f"Audit Ready: {readiness:.1f}%")
        """
        if not assessment.criteria:
            return 0.0

        compliant_count = sum(
            1 for c in assessment.criteria
            if c.score >= ScoreLevel.COMPLIANT
        )

        readiness = (compliant_count / len(assessment.criteria)) * 100
        return round(readiness, 2)

    def get_tested_percentage(self, assessment: Assessment) -> float:
        """Calculate percentage of tested controls.

        Measures what percentage of criteria have achieved at least
        TESTED (3) status.

        Args:
            assessment: The assessment to evaluate.

        Returns:
            Percentage of criteria at TESTED level or above (0-100).
        """
        if not assessment.criteria:
            return 0.0

        tested_count = sum(
            1 for c in assessment.criteria
            if c.score >= ScoreLevel.TESTED
        )

        return round((tested_count / len(assessment.criteria)) * 100, 2)

    def get_implemented_percentage(self, assessment: Assessment) -> float:
        """Calculate percentage of implemented controls.

        Measures what percentage of criteria have achieved at least
        IMPLEMENTED (2) status.

        Args:
            assessment: The assessment to evaluate.

        Returns:
            Percentage of criteria at IMPLEMENTED level or above (0-100).
        """
        if not assessment.criteria:
            return 0.0

        implemented_count = sum(
            1 for c in assessment.criteria
            if c.score >= ScoreLevel.IMPLEMENTED
        )

        return round((implemented_count / len(assessment.criteria)) * 100, 2)

    def score_to_status(self, score: float) -> str:
        """Convert numeric score to status string.

        Interprets the overall score using defined thresholds.

        Args:
            score: The overall score (0-100).

        Returns:
            Status string: "not_ready", "partial", "ready", "audit_ready",
            or "compliant".

        Example:
            >>> status = scorer.score_to_status(78.5)
            >>> print(status)
            'audit_ready'
        """
        if score < self.thresholds.not_ready_max:
            return "not_ready"
        elif score < self.thresholds.partial_max:
            return "partial"
        elif score < self.thresholds.ready_max:
            return "ready"
        elif score < self.thresholds.compliant_min:
            return "audit_ready"
        else:
            return "compliant"

    def score_to_label(self, score: float) -> str:
        """Convert numeric score to human-readable label.

        Args:
            score: The overall score (0-100).

        Returns:
            Human-readable status label.
        """
        status = self.score_to_status(score)
        labels = {
            "not_ready": "Not Ready - Significant gaps exist",
            "partial": "Partial Readiness - Major improvements needed",
            "ready": "Ready with Gaps - Address gaps before audit",
            "audit_ready": "Audit Ready - Minor improvements recommended",
            "compliant": "Fully Compliant - Ready for Type II audit",
        }
        return labels.get(status, "Unknown Status")

    def get_score_distribution(
        self,
        assessment: Assessment,
    ) -> Dict[str, int]:
        """Get distribution of criteria across maturity levels.

        Args:
            assessment: The assessment to analyze.

        Returns:
            Dictionary mapping maturity level names to counts.

        Example:
            >>> dist = scorer.get_score_distribution(assessment)
            >>> print(dist)
            {'NOT_IMPLEMENTED': 5, 'PARTIAL': 10, 'IMPLEMENTED': 15, ...}
        """
        distribution = {level.name: 0 for level in MaturityLevel}

        for criterion in assessment.criteria:
            try:
                level = MaturityLevel(criterion.score)
                distribution[level.name] += 1
            except ValueError:
                # Handle any unexpected score values
                distribution["NOT_IMPLEMENTED"] += 1

        return distribution

    def calculate_gap_severity_score(
        self,
        assessment: Assessment,
    ) -> float:
        """Calculate severity-weighted gap score.

        Weights gaps by criterion risk level to prioritize critical gaps.

        Args:
            assessment: The assessment to analyze.

        Returns:
            Severity-weighted gap score (lower is better).
        """
        risk_weights = {
            "critical": 4.0,
            "high": 3.0,
            "medium": 2.0,
            "low": 1.0,
        }

        total_gap_weight = 0.0
        max_possible_weight = 0.0

        for criterion in assessment.criteria:
            criterion_def = TSC_CRITERIA.get(criterion.criterion_id, {})
            risk_level = criterion_def.get("risk_level", "medium")
            weight = risk_weights.get(risk_level, 2.0)

            # Gap = (4 - actual_score) * weight
            gap = (4 - criterion.score) * weight
            total_gap_weight += gap
            max_possible_weight += 4 * weight

        if max_possible_weight == 0:
            return 0.0

        # Normalize to 0-100 (inverted so higher = worse)
        return round((total_gap_weight / max_possible_weight) * 100, 2)

    def get_critical_gaps(
        self,
        assessment: Assessment,
        max_score: int = ScoreLevel.PARTIAL,
    ) -> List[Tuple[str, int, str]]:
        """Get list of critical and high-risk gaps.

        Identifies criteria with low scores that have critical or high
        risk levels.

        Args:
            assessment: The assessment to analyze.
            max_score: Maximum score to be considered a gap.

        Returns:
            List of tuples: (criterion_id, score, risk_level).
        """
        gaps = []

        for criterion in assessment.criteria:
            if criterion.score <= max_score:
                criterion_def = TSC_CRITERIA.get(criterion.criterion_id, {})
                risk_level = criterion_def.get("risk_level", "medium")

                if risk_level in ("critical", "high"):
                    gaps.append((
                        criterion.criterion_id,
                        criterion.score,
                        risk_level,
                    ))

        # Sort by risk level (critical first) then by score (lowest first)
        risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        gaps.sort(key=lambda x: (risk_order.get(x[2], 2), x[1]))

        return gaps

    def calculate_trend(
        self,
        current: Assessment,
        previous: Assessment,
    ) -> Dict[str, float]:
        """Calculate score trend between two assessments.

        Args:
            current: The current assessment.
            previous: The previous assessment for comparison.

        Returns:
            Dictionary with trend metrics (delta, percentage change).
        """
        current_score = self.calculate_overall_score(current)
        previous_score = self.calculate_overall_score(previous)

        delta = current_score - previous_score
        if previous_score > 0:
            pct_change = (delta / previous_score) * 100
        else:
            pct_change = 100.0 if current_score > 0 else 0.0

        return {
            "current_score": current_score,
            "previous_score": previous_score,
            "delta": round(delta, 2),
            "percentage_change": round(pct_change, 2),
            "trend": "improving" if delta > 0 else "declining" if delta < 0 else "stable",
        }


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def calculate_score(assessment: Assessment) -> float:
    """Calculate overall score for an assessment.

    Convenience function for quick scoring without instantiating Scorer.

    Args:
        assessment: The assessment to score.

    Returns:
        Overall score (0-100).
    """
    scorer = Scorer()
    return scorer.calculate_overall_score(assessment)


def get_readiness(assessment: Assessment) -> float:
    """Get audit readiness percentage.

    Convenience function for quick readiness check.

    Args:
        assessment: The assessment to evaluate.

    Returns:
        Readiness percentage (0-100).
    """
    scorer = Scorer()
    return scorer.get_readiness_percentage(assessment)


def score_to_status(score: float) -> str:
    """Convert score to status string.

    Convenience function for quick status determination.

    Args:
        score: Overall score (0-100).

    Returns:
        Status string.
    """
    scorer = Scorer()
    return scorer.score_to_status(score)


__all__ = [
    "Scorer",
    "MaturityLevel",
    "ScoreThresholds",
    "DEFAULT_THRESHOLDS",
    "calculate_score",
    "get_readiness",
    "score_to_status",
]
