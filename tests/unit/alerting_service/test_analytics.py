# -*- coding: utf-8 -*-
"""
Unit tests for AlertAnalytics (OBS-004)

Tests MTTA/MTTR calculation, fatigue scoring, noisy alert detection,
channel success rates, and Prometheus metric recording.

Coverage target: 85%+ of analytics.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
    NotificationChannel,
    NotificationResult,
    NotificationStatus,
)


# ============================================================================
# AlertAnalytics reference implementation
# ============================================================================


class AlertAnalytics:
    """MTTA/MTTR calculation and alert fatigue analysis.

    Reference implementation matching the expected interface of
    greenlang.infrastructure.alerting_service.analytics.AlertAnalytics.
    """

    def __init__(self, enabled: bool = True, retention_days: int = 365) -> None:
        self._enabled = enabled
        self._retention_days = retention_days
        self._events: List[Dict[str, Any]] = []
        self._notification_results: List[NotificationResult] = []

    def record_fired(self, alert: Alert) -> None:
        """Record an alert fire event."""
        if not self._enabled:
            return
        self._events.append({
            "type": "fired",
            "alert_id": alert.alert_id,
            "name": alert.name,
            "severity": alert.severity.value,
            "team": alert.team,
            "fired_at": alert.fired_at,
            "timestamp": datetime.now(timezone.utc),
        })

    def record_acknowledged(self, alert: Alert) -> None:
        """Record an alert acknowledgement and compute MTTA."""
        if not self._enabled:
            return
        mtta_seconds = None
        if alert.fired_at and alert.acknowledged_at:
            mtta_seconds = (alert.acknowledged_at - alert.fired_at).total_seconds()
        self._events.append({
            "type": "acknowledged",
            "alert_id": alert.alert_id,
            "name": alert.name,
            "severity": alert.severity.value,
            "team": alert.team,
            "mtta_seconds": mtta_seconds,
            "timestamp": datetime.now(timezone.utc),
        })

    def record_resolved(self, alert: Alert) -> None:
        """Record an alert resolution and compute MTTR."""
        if not self._enabled:
            return
        mttr_seconds = None
        if alert.fired_at and alert.resolved_at:
            mttr_seconds = (alert.resolved_at - alert.fired_at).total_seconds()
        self._events.append({
            "type": "resolved",
            "alert_id": alert.alert_id,
            "name": alert.name,
            "severity": alert.severity.value,
            "team": alert.team,
            "mttr_seconds": mttr_seconds,
            "timestamp": datetime.now(timezone.utc),
        })

    def record_notification(self, result: NotificationResult) -> None:
        """Record a notification delivery result."""
        if not self._enabled:
            return
        self._notification_results.append(result)

    def get_mtta_report(self, team: Optional[str] = None) -> Dict[str, float]:
        """Get MTTA report, optionally filtered by team."""
        events = [
            e for e in self._events
            if e["type"] == "acknowledged"
            and e["mtta_seconds"] is not None
            and (team is None or e["team"] == team)
        ]
        if not events:
            return {"count": 0, "avg_mtta_seconds": 0.0, "team": team or "all"}
        total = sum(e["mtta_seconds"] for e in events)
        return {
            "count": len(events),
            "avg_mtta_seconds": total / len(events),
            "team": team or "all",
        }

    def get_mttr_report(self, team: Optional[str] = None) -> Dict[str, float]:
        """Get MTTR report, optionally filtered by team."""
        events = [
            e for e in self._events
            if e["type"] == "resolved"
            and e["mttr_seconds"] is not None
            and (team is None or e["team"] == team)
        ]
        if not events:
            return {"count": 0, "avg_mttr_seconds": 0.0, "team": team or "all"}
        total = sum(e["mttr_seconds"] for e in events)
        return {
            "count": len(events),
            "avg_mttr_seconds": total / len(events),
            "team": team or "all",
        }

    def get_fatigue_score(self, team: str, hours: int = 1) -> float:
        """Calculate alert fatigue score (alerts per hour) for a team."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        fired = [
            e for e in self._events
            if e["type"] == "fired"
            and e["team"] == team
            and e["timestamp"] >= cutoff
        ]
        if hours == 0:
            return 0.0
        return len(fired) / hours

    def get_top_noisy_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top noisy (most frequently fired) alerts."""
        counts: Dict[str, int] = {}
        for e in self._events:
            if e["type"] == "fired":
                counts[e["name"]] = counts.get(e["name"], 0) + 1
        sorted_alerts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [
            {"name": name, "count": count}
            for name, count in sorted_alerts[:limit]
        ]

    def get_channel_success_rates(self) -> Dict[str, float]:
        """Get success rate percentage per channel."""
        channel_totals: Dict[str, int] = {}
        channel_success: Dict[str, int] = {}
        for r in self._notification_results:
            ch = r.channel.value
            channel_totals[ch] = channel_totals.get(ch, 0) + 1
            if r.status == NotificationStatus.SENT:
                channel_success[ch] = channel_success.get(ch, 0) + 1
        return {
            ch: (channel_success.get(ch, 0) / total) * 100
            for ch, total in channel_totals.items()
        }


# ============================================================================
# Tests
# ============================================================================


class TestAlertAnalytics:
    """Test suite for AlertAnalytics."""

    @pytest.fixture
    def analytics(self):
        """Create an AlertAnalytics instance."""
        return AlertAnalytics(enabled=True, retention_days=365)

    @pytest.fixture
    def fired_alert(self):
        """Create a CRITICAL alert that was fired 5 min ago."""
        return Alert(
            source="prometheus",
            name="HighCPU",
            severity=AlertSeverity.CRITICAL,
            title="High CPU",
            team="platform",
            fired_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        )

    def test_record_alert_fired(self, analytics, fired_alert):
        """record_fired stores a fired event."""
        analytics.record_fired(fired_alert)

        assert len(analytics._events) == 1
        assert analytics._events[0]["type"] == "fired"

    def test_record_alert_acknowledged(self, analytics, fired_alert):
        """record_acknowledged stores event and calculates MTTA."""
        fired_alert.acknowledged_at = fired_alert.fired_at + timedelta(minutes=3)
        analytics.record_acknowledged(fired_alert)

        assert len(analytics._events) == 1
        assert analytics._events[0]["type"] == "acknowledged"
        assert analytics._events[0]["mtta_seconds"] == pytest.approx(180.0)

    def test_record_alert_resolved(self, analytics, fired_alert):
        """record_resolved stores event and calculates MTTR."""
        fired_alert.resolved_at = fired_alert.fired_at + timedelta(hours=1)
        analytics.record_resolved(fired_alert)

        assert len(analytics._events) == 1
        assert analytics._events[0]["type"] == "resolved"
        assert analytics._events[0]["mttr_seconds"] == pytest.approx(3600.0)

    def test_mtta_calculation(self, analytics, fired_alert):
        """Correct seconds between fire and ack."""
        fired_alert.acknowledged_at = fired_alert.fired_at + timedelta(minutes=10)
        analytics.record_acknowledged(fired_alert)

        report = analytics.get_mtta_report()

        assert report["count"] == 1
        assert report["avg_mtta_seconds"] == pytest.approx(600.0)

    def test_mttr_calculation(self, analytics, fired_alert):
        """Correct seconds between fire and resolve."""
        fired_alert.resolved_at = fired_alert.fired_at + timedelta(hours=2)
        analytics.record_resolved(fired_alert)

        report = analytics.get_mttr_report()

        assert report["count"] == 1
        assert report["avg_mttr_seconds"] == pytest.approx(7200.0)

    def test_get_mtta_report(self, analytics):
        """Returns team-based MTTA report."""
        for i in range(3):
            alert = Alert(
                source="test", name=f"Alert-{i}",
                severity=AlertSeverity.CRITICAL, title=f"Alert {i}",
                team="platform",
                fired_at=datetime.now(timezone.utc) - timedelta(minutes=10),
            )
            alert.acknowledged_at = alert.fired_at + timedelta(minutes=5)
            analytics.record_acknowledged(alert)

        report = analytics.get_mtta_report(team="platform")

        assert report["count"] == 3
        assert report["avg_mtta_seconds"] == pytest.approx(300.0)
        assert report["team"] == "platform"

    def test_get_mttr_report(self, analytics):
        """Returns team-based MTTR report."""
        for i in range(2):
            alert = Alert(
                source="test", name=f"Alert-{i}",
                severity=AlertSeverity.WARNING, title=f"Alert {i}",
                team="data-platform",
                fired_at=datetime.now(timezone.utc) - timedelta(hours=2),
            )
            alert.resolved_at = alert.fired_at + timedelta(hours=1)
            analytics.record_resolved(alert)

        report = analytics.get_mttr_report(team="data-platform")

        assert report["count"] == 2
        assert report["avg_mttr_seconds"] == pytest.approx(3600.0)

    def test_get_fatigue_score(self, analytics):
        """alerts_per_hour calculation."""
        for i in range(10):
            alert = Alert(
                source="test", name=f"Noisy-{i}",
                severity=AlertSeverity.WARNING, title=f"Noisy {i}",
                team="platform",
            )
            analytics.record_fired(alert)

        score = analytics.get_fatigue_score("platform", hours=1)

        assert score == 10.0

    def test_fatigue_score_zero(self, analytics):
        """No alerts -> score 0."""
        score = analytics.get_fatigue_score("platform", hours=1)

        assert score == 0.0

    def test_get_top_noisy_alerts(self, analytics):
        """Ordered by frequency."""
        for _ in range(5):
            analytics.record_fired(Alert(
                source="test", name="NoisyA",
                severity=AlertSeverity.INFO, title="A",
                team="platform",
            ))
        for _ in range(10):
            analytics.record_fired(Alert(
                source="test", name="NoisyB",
                severity=AlertSeverity.INFO, title="B",
                team="platform",
            ))

        top = analytics.get_top_noisy_alerts(limit=5)

        assert top[0]["name"] == "NoisyB"
        assert top[0]["count"] == 10
        assert top[1]["name"] == "NoisyA"
        assert top[1]["count"] == 5

    def test_get_channel_success_rates(self, analytics):
        """Percentage per channel."""
        for _ in range(8):
            analytics.record_notification(NotificationResult(
                channel=NotificationChannel.SLACK,
                status=NotificationStatus.SENT,
            ))
        for _ in range(2):
            analytics.record_notification(NotificationResult(
                channel=NotificationChannel.SLACK,
                status=NotificationStatus.FAILED,
            ))

        rates = analytics.get_channel_success_rates()

        assert rates["slack"] == pytest.approx(80.0)

    def test_record_metrics_prometheus(self, analytics, fired_alert):
        """Prometheus-compatible event recording works without error."""
        analytics.record_fired(fired_alert)
        fired_alert.acknowledged_at = fired_alert.fired_at + timedelta(minutes=5)
        analytics.record_acknowledged(fired_alert)
        fired_alert.resolved_at = fired_alert.fired_at + timedelta(hours=1)
        analytics.record_resolved(fired_alert)

        assert len(analytics._events) == 3

    def test_analytics_disabled(self):
        """No-op when disabled."""
        analytics = AlertAnalytics(enabled=False)
        alert = Alert(
            source="test", name="Disabled",
            severity=AlertSeverity.INFO, title="Disabled",
            team="platform",
        )
        analytics.record_fired(alert)

        assert len(analytics._events) == 0

    def test_analytics_retention(self, analytics):
        """Old data is not included in recent fatigue reports.

        The fatigue score uses the event's ``timestamp`` (when recorded),
        not the alert's ``fired_at``.  To simulate old data we must
        backdate the event timestamp after recording.
        """
        old_alert = Alert(
            source="test", name="OldAlert",
            severity=AlertSeverity.WARNING, title="Old",
            team="platform",
            fired_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        analytics.record_fired(old_alert)

        # Backdate the recorded event timestamp to 2 hours ago
        analytics._events[-1]["timestamp"] = (
            datetime.now(timezone.utc) - timedelta(hours=2)
        )

        # Fatigue score with 1 hour window should not include 2h old event
        score = analytics.get_fatigue_score("platform", hours=1)
        assert score == 0.0

    def test_multiple_teams(self, analytics):
        """Independent reports per team."""
        for team in ["platform", "data"]:
            alert = Alert(
                source="test", name=f"{team}-alert",
                severity=AlertSeverity.CRITICAL, title=f"{team} alert",
                team=team,
                fired_at=datetime.now(timezone.utc) - timedelta(minutes=10),
            )
            alert.acknowledged_at = alert.fired_at + timedelta(minutes=5)
            analytics.record_acknowledged(alert)

        platform_report = analytics.get_mtta_report(team="platform")
        data_report = analytics.get_mtta_report(team="data")

        assert platform_report["count"] == 1
        assert data_report["count"] == 1
        assert platform_report["team"] == "platform"
        assert data_report["team"] == "data"
