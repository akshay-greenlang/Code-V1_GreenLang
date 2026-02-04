# -*- coding: utf-8 -*-
"""
Feature Flag Stale Detector - INFRA-008

Detects feature flags that have not been evaluated or modified recently
and may be candidates for cleanup or archival. Provides health reporting
for the overall flag inventory.

Stale flags accumulate technical debt and can:
    - Confuse developers about active feature state
    - Slow down flag evaluation (larger flag sets)
    - Create compliance risks (orphaned configurations)

The detector uses the flag's ``updated_at`` timestamp as the primary
staleness signal. In production environments with PostgreSQL metrics tables,
this would be augmented with actual evaluation timestamps.

Example:
    >>> from greenlang.infrastructure.feature_flags.lifecycle.stale_detector import StaleFlagDetector
    >>> detector = StaleFlagDetector(service)
    >>> stale_flags = await detector.detect_stale_flags(days_threshold=30)
    >>> report = await detector.get_flag_health_report()
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.feature_flags.models import (
    FeatureFlag,
    FlagStatus,
    FlagType,
)

logger = logging.getLogger(__name__)


class StaleFlagDetector:
    """Detects stale feature flags and generates health reports.

    A flag is considered stale if it has not been updated within the
    specified threshold and is not in a terminal state (ARCHIVED,
    PERMANENT). Stale flags are candidates for review, archival,
    or promotion to permanent.

    Attributes:
        _service: The FeatureFlagService for flag access.
    """

    def __init__(self, service: "FeatureFlagService") -> None:  # noqa: F821
        """Initialize the stale flag detector.

        Args:
            service: The FeatureFlagService instance.
        """
        # Import here to avoid circular imports at module level
        from greenlang.infrastructure.feature_flags.service import FeatureFlagService

        self._service: FeatureFlagService = service
        logger.info("StaleFlagDetector initialized")

    async def detect_stale_flags(
        self,
        days_threshold: int = 30,
    ) -> List[FeatureFlag]:
        """Find flags that have not been updated within the threshold.

        Flags in ARCHIVED or PERMANENT status are excluded from stale
        detection because they have explicit lifecycle states.

        Args:
            days_threshold: Number of days without update to consider
                a flag stale. Defaults to 30 days.

        Returns:
            List of FeatureFlag instances that are stale, sorted by
            staleness (most stale first).
        """
        all_flags = await self._service.list_flags(offset=0, limit=10000)
        now = datetime.now(timezone.utc)
        threshold_seconds = days_threshold * 86400
        stale_flags: List[FeatureFlag] = []

        for flag in all_flags:
            # Skip terminal states
            if flag.status in (FlagStatus.ARCHIVED, FlagStatus.PERMANENT):
                continue

            age_seconds = (now - flag.updated_at).total_seconds()
            if age_seconds >= threshold_seconds:
                stale_flags.append(flag)

        # Sort by staleness (most stale first)
        stale_flags.sort(key=lambda f: f.updated_at)

        logger.info(
            "Detected %d stale flags (threshold=%d days, total=%d)",
            len(stale_flags), days_threshold, len(all_flags),
        )
        return stale_flags

    async def get_flag_health_report(self) -> Dict[str, Any]:
        """Generate a comprehensive health report for all feature flags.

        The report includes:
            - Total flag count and breakdown by status/type
            - Stale flag count at multiple thresholds
            - Age distribution statistics
            - Flags with no evaluations (based on updated_at heuristic)
            - Actionable recommendations

        Returns:
            Dictionary containing the health report.
        """
        all_flags = await self._service.list_flags(offset=0, limit=10000)
        now = datetime.now(timezone.utc)

        # Status distribution
        status_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        for flag in all_flags:
            status_counts[flag.status.value] = (
                status_counts.get(flag.status.value, 0) + 1
            )
            type_counts[flag.flag_type.value] = (
                type_counts.get(flag.flag_type.value, 0) + 1
            )

        # Age analysis for non-terminal flags
        active_flags = [
            f for f in all_flags
            if f.status not in (FlagStatus.ARCHIVED, FlagStatus.PERMANENT)
        ]

        ages_days = [
            (now - f.updated_at).days for f in active_flags
        ]

        # Stale at multiple thresholds
        stale_7d = len([a for a in ages_days if a >= 7])
        stale_14d = len([a for a in ages_days if a >= 14])
        stale_30d = len([a for a in ages_days if a >= 30])
        stale_60d = len([a for a in ages_days if a >= 60])
        stale_90d = len([a for a in ages_days if a >= 90])

        # Age statistics
        avg_age = sum(ages_days) / len(ages_days) if ages_days else 0.0
        max_age = max(ages_days) if ages_days else 0
        min_age = min(ages_days) if ages_days else 0

        # Killed flags that should be reviewed
        killed_flags = [
            f for f in all_flags if f.status == FlagStatus.KILLED
        ]
        killed_details = [
            {
                "key": f.key,
                "killed_age_days": (now - f.updated_at).days,
                "owner": f.owner,
            }
            for f in killed_flags
        ]

        # Flags at 100% rollout that are still ACTIVE
        full_rollout_active = [
            f for f in all_flags
            if f.status == FlagStatus.ACTIVE and f.rollout_percentage >= 100.0
        ]

        # Build recommendations
        recommendations = self._build_recommendations(
            stale_30d=stale_30d,
            killed_count=len(killed_flags),
            full_rollout_active_count=len(full_rollout_active),
            total_active=len(active_flags),
        )

        report = {
            "timestamp": now.isoformat(),
            "total_flags": len(all_flags),
            "active_flags": len(active_flags),
            "by_status": status_counts,
            "by_type": type_counts,
            "stale_flags": {
                "7_days": stale_7d,
                "14_days": stale_14d,
                "30_days": stale_30d,
                "60_days": stale_60d,
                "90_days": stale_90d,
            },
            "age_statistics": {
                "average_days": round(avg_age, 1),
                "max_days": max_age,
                "min_days": min_age,
            },
            "killed_flags": killed_details,
            "full_rollout_active": [
                {"key": f.key, "owner": f.owner, "age_days": (now - f.updated_at).days}
                for f in full_rollout_active
            ],
            "health_score": self._calculate_health_score(
                total=len(all_flags),
                stale_30d=stale_30d,
                killed=len(killed_flags),
                archived=status_counts.get("archived", 0),
            ),
            "recommendations": recommendations,
        }

        logger.info(
            "Flag health report generated: %d total, %d stale (30d), score=%.1f%%",
            report["total_flags"],
            stale_30d,
            report["health_score"],
        )
        return report

    @staticmethod
    def _calculate_health_score(
        total: int,
        stale_30d: int,
        killed: int,
        archived: int,
    ) -> float:
        """Calculate a health score for the flag inventory.

        The score is a percentage (0-100) where higher is healthier.
        Deductions are applied for:
            - Stale flags (30d): -2 points each
            - Killed flags: -5 points each
            - High ratio of archived flags: no deduction (cleanup is good)

        Args:
            total: Total number of flags.
            stale_30d: Number of stale flags (30-day threshold).
            killed: Number of killed flags.
            archived: Number of archived flags.

        Returns:
            Health score as a float (0.0 to 100.0).
        """
        if total == 0:
            return 100.0

        active_non_archived = total - archived
        if active_non_archived == 0:
            return 100.0

        score = 100.0

        # Deductions for stale flags
        stale_ratio = stale_30d / active_non_archived
        score -= stale_ratio * 40  # Max 40-point deduction

        # Deductions for killed flags
        killed_ratio = killed / active_non_archived
        score -= killed_ratio * 30  # Max 30-point deduction

        return max(0.0, min(100.0, round(score, 1)))

    @staticmethod
    def _build_recommendations(
        stale_30d: int,
        killed_count: int,
        full_rollout_active_count: int,
        total_active: int,
    ) -> List[str]:
        """Build actionable recommendations based on flag health.

        Args:
            stale_30d: Number of stale flags at 30-day threshold.
            killed_count: Number of killed flags.
            full_rollout_active_count: Flags at 100% rollout still ACTIVE.
            total_active: Total number of active (non-terminal) flags.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        if stale_30d > 0:
            recommendations.append(
                f"Review {stale_30d} stale flag(s) that have not been "
                f"updated in 30+ days. Consider archiving or promoting."
            )

        if killed_count > 0:
            recommendations.append(
                f"Review {killed_count} killed flag(s). Consider "
                f"archiving flags that will not be restored."
            )

        if full_rollout_active_count > 0:
            recommendations.append(
                f"Promote {full_rollout_active_count} flag(s) at 100% "
                f"rollout from ACTIVE to ROLLED_OUT or PERMANENT."
            )

        if total_active > 500:
            recommendations.append(
                f"Flag inventory ({total_active} active flags) is large. "
                f"Consider a cleanup sprint to reduce cognitive load."
            )

        if not recommendations:
            recommendations.append(
                "Flag inventory is healthy. No immediate action needed."
            )

        return recommendations


__all__ = ["StaleFlagDetector"]
