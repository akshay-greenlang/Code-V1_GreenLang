# -*- coding: utf-8 -*-
"""
Alert Analytics - OBS-004: Unified Alerting Service

Tracks and reports on operational alerting metrics including Mean Time
To Acknowledge (MTTA), Mean Time To Resolve (MTTR), alert fatigue
scoring, channel success rates, and noisy-alert ranking.

Example:
    >>> analytics = AlertAnalytics(config)
    >>> analytics.record_alert_fired(alert)
    >>> analytics.record_alert_acknowledged(alert)
    >>> report = analytics.get_mtta_report("platform", period_hours=24)

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.infrastructure.alerting_service.config import AlertingConfig
from greenlang.infrastructure.alerting_service.metrics import (
    record_mtta,
    record_mttr,
    update_fatigue_score,
)
from greenlang.infrastructure.alerting_service.models import Alert

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal metric entry
# ---------------------------------------------------------------------------


@staticmethod
def _now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# AlertAnalytics
# ---------------------------------------------------------------------------


class AlertAnalytics:
    """Analytics engine for alerting operations.

    Maintains in-memory time-series data for MTTA, MTTR, alert volumes,
    and channel delivery outcomes. Data is retained for the configured
    retention period.

    Attributes:
        config: AlertingConfig instance.
    """

    def __init__(self, config: AlertingConfig) -> None:
        self.config = config
        self._lock = threading.RLock()

        # Per-team MTTA/MTTR observations: team -> [(timestamp, seconds)]
        self._mtta_observations: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self._mttr_observations: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)

        # Alert volume: team -> [fired_at timestamps]
        self._alert_volumes: Dict[str, List[datetime]] = defaultdict(list)

        # Alert name frequency: name -> count
        self._alert_name_counts: Dict[str, int] = defaultdict(int)

        # Channel delivery: channel -> {sent: int, failed: int}
        self._channel_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"sent": 0, "failed": 0},
        )

        logger.info(
            "AlertAnalytics initialized: retention=%dd",
            config.analytics_retention_days,
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_alert_fired(self, alert: Alert) -> None:
        """Record an alert firing event.

        Args:
            alert: The fired alert.
        """
        now = datetime.now(timezone.utc)
        team = alert.team or "unassigned"
        with self._lock:
            self._alert_volumes[team].append(now)
            self._alert_name_counts[alert.name] += 1

    def record_alert_acknowledged(self, alert: Alert) -> None:
        """Record an alert acknowledgement and compute MTTA.

        Args:
            alert: The acknowledged alert (must have ``acknowledged_at``).
        """
        if not alert.fired_at or not alert.acknowledged_at:
            return

        mtta_seconds = (alert.acknowledged_at - alert.fired_at).total_seconds()
        team = alert.team or "unassigned"

        with self._lock:
            self._mtta_observations[team].append(
                (datetime.now(timezone.utc), mtta_seconds),
            )

        record_mtta(team, alert.severity.value, mtta_seconds)
        logger.debug(
            "MTTA recorded: alert=%s, team=%s, mtta=%.1fs",
            alert.alert_id[:8], team, mtta_seconds,
        )

    def record_alert_resolved(self, alert: Alert) -> None:
        """Record an alert resolution and compute MTTR.

        Args:
            alert: The resolved alert (must have ``resolved_at``).
        """
        if not alert.fired_at or not alert.resolved_at:
            return

        mttr_seconds = (alert.resolved_at - alert.fired_at).total_seconds()
        team = alert.team or "unassigned"

        with self._lock:
            self._mttr_observations[team].append(
                (datetime.now(timezone.utc), mttr_seconds),
            )

        record_mttr(team, alert.severity.value, mttr_seconds)
        logger.debug(
            "MTTR recorded: alert=%s, team=%s, mttr=%.1fs",
            alert.alert_id[:8], team, mttr_seconds,
        )

    def record_notification_result(
        self,
        channel: str,
        success: bool,
    ) -> None:
        """Record a notification delivery outcome.

        Args:
            channel: Channel name.
            success: True if delivered successfully.
        """
        key = "sent" if success else "failed"
        with self._lock:
            self._channel_stats[channel][key] += 1

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def get_mtta_report(
        self,
        team: str = "",
        period_hours: int = 24,
    ) -> Dict[str, Any]:
        """Compute MTTA statistics for a team over a time period.

        Args:
            team: Team name (empty for all teams).
            period_hours: Look-back window in hours.

        Returns:
            Dict with ``mean_seconds``, ``p50_seconds``, ``p95_seconds``,
            ``sample_count``, ``period_hours``.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=period_hours)
        return self._compute_report(self._mtta_observations, team, cutoff, period_hours)

    def get_mttr_report(
        self,
        team: str = "",
        period_hours: int = 24,
    ) -> Dict[str, Any]:
        """Compute MTTR statistics for a team over a time period.

        Args:
            team: Team name (empty for all teams).
            period_hours: Look-back window in hours.

        Returns:
            Dict with ``mean_seconds``, ``p50_seconds``, ``p95_seconds``,
            ``sample_count``, ``period_hours``.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=period_hours)
        return self._compute_report(self._mttr_observations, team, cutoff, period_hours)

    def get_fatigue_score(self, team: str = "") -> float:
        """Compute alert fatigue score (alerts per hour) for a team.

        Uses the last 4 hours of data. A score above 10 indicates high
        fatigue risk.

        Args:
            team: Team name (empty for all teams).

        Returns:
            Alerts per hour as a float.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=4)
        count = 0

        with self._lock:
            if team:
                volumes = self._alert_volumes.get(team, [])
                count = sum(1 for t in volumes if t > cutoff)
            else:
                for vol_list in self._alert_volumes.values():
                    count += sum(1 for t in vol_list if t > cutoff)

        score = count / 4.0  # per hour over 4h window
        if team:
            update_fatigue_score(team, score)
        return score

    def get_top_noisy_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the most frequently firing alert names.

        Args:
            limit: Maximum results.

        Returns:
            List of dicts with ``name`` and ``count``.
        """
        with self._lock:
            sorted_names = sorted(
                self._alert_name_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        return [
            {"name": name, "count": count}
            for name, count in sorted_names[:limit]
        ]

    def get_channel_success_rates(self) -> Dict[str, float]:
        """Compute delivery success rate per channel.

        Returns:
            Dict mapping channel name to success rate (0.0-1.0).
        """
        rates: Dict[str, float] = {}
        with self._lock:
            for channel, stats in self._channel_stats.items():
                total = stats["sent"] + stats["failed"]
                if total == 0:
                    rates[channel] = 1.0
                else:
                    rates[channel] = stats["sent"] / total
        return rates

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> int:
        """Remove observations older than the retention period.

        Returns:
            Number of observations removed.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=self.config.analytics_retention_days,
        )
        removed = 0

        with self._lock:
            for team in list(self._mtta_observations.keys()):
                before = len(self._mtta_observations[team])
                self._mtta_observations[team] = [
                    obs for obs in self._mtta_observations[team] if obs[0] > cutoff
                ]
                removed += before - len(self._mtta_observations[team])

            for team in list(self._mttr_observations.keys()):
                before = len(self._mttr_observations[team])
                self._mttr_observations[team] = [
                    obs for obs in self._mttr_observations[team] if obs[0] > cutoff
                ]
                removed += before - len(self._mttr_observations[team])

            for team in list(self._alert_volumes.keys()):
                before = len(self._alert_volumes[team])
                self._alert_volumes[team] = [
                    t for t in self._alert_volumes[team] if t > cutoff
                ]
                removed += before - len(self._alert_volumes[team])

        if removed:
            logger.info("Analytics cleanup: removed %d old observations", removed)
        return removed

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_report(
        self,
        observations: Dict[str, List[Tuple[datetime, float]]],
        team: str,
        cutoff: datetime,
        period_hours: int,
    ) -> Dict[str, Any]:
        """Compute statistical summary over observations.

        Args:
            observations: Team-keyed observation lists.
            team: Filter team (empty for all).
            cutoff: Only include observations after this time.
            period_hours: For the report metadata.

        Returns:
            Report dict.
        """
        values: List[float] = []
        with self._lock:
            if team:
                for ts, val in observations.get(team, []):
                    if ts > cutoff:
                        values.append(val)
            else:
                for obs_list in observations.values():
                    for ts, val in obs_list:
                        if ts > cutoff:
                            values.append(val)

        if not values:
            return {
                "mean_seconds": 0.0,
                "p50_seconds": 0.0,
                "p95_seconds": 0.0,
                "sample_count": 0,
                "period_hours": period_hours,
            }

        values.sort()
        n = len(values)
        mean_val = sum(values) / n
        p50_idx = int(n * 0.5)
        p95_idx = min(int(n * 0.95), n - 1)

        return {
            "mean_seconds": round(mean_val, 2),
            "p50_seconds": round(values[p50_idx], 2),
            "p95_seconds": round(values[p95_idx], 2),
            "sample_count": n,
            "period_hours": period_hours,
        }
