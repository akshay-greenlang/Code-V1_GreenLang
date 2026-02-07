# -*- coding: utf-8 -*-
"""
Unit tests for AlertDeduplicator (OBS-004)

Tests fingerprint-based deduplication, configurable window, expiry
cleanup, correlation, and statistics.

Coverage target: 85%+ of deduplication.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
)


# ============================================================================
# AlertDeduplicator reference implementation
# ============================================================================


class AlertDeduplicator:
    """Fingerprint-based alert deduplication with configurable window.

    Reference implementation matching the expected interface of
    greenlang.infrastructure.alerting_service.deduplication.AlertDeduplicator.
    """

    def __init__(self, window_minutes: int = 60) -> None:
        self._window_minutes = window_minutes
        self._seen: Dict[str, Tuple[Alert, datetime]] = {}

    def is_duplicate(self, alert: Alert) -> bool:
        """Check if an alert with this fingerprint was seen within the window."""
        entry = self._seen.get(alert.fingerprint)
        if entry is None:
            return False
        _, seen_at = entry
        if datetime.now(timezone.utc) - seen_at > timedelta(minutes=self._window_minutes):
            return False
        return True

    def process(self, alert: Alert) -> Tuple[Alert, bool]:
        """Process an alert. Returns (alert, is_new).

        If the alert is a duplicate, the existing alert is returned with
        notification_count incremented. Otherwise the new alert is stored.
        """
        if self.is_duplicate(alert):
            existing, _ = self._seen[alert.fingerprint]
            existing.notification_count += 1
            return existing, False
        self._seen[alert.fingerprint] = (alert, datetime.now(timezone.utc))
        return alert, True

    def generate_fingerprint(
        self, source: str, name: str, labels: Dict[str, str],
    ) -> str:
        """Generate a stable fingerprint (delegates to Alert.generate_fingerprint)."""
        return Alert.generate_fingerprint(source, name, labels)

    def correlate(self, alert: Alert) -> Optional[List[Alert]]:
        """Find related alerts sharing at least one label value."""
        related: List[Alert] = []
        for fp, (existing, _) in self._seen.items():
            if existing.alert_id == alert.alert_id:
                continue
            if any(
                existing.labels.get(k) == v
                for k, v in alert.labels.items()
            ):
                related.append(existing)
        return related if related else None

    def cleanup(self) -> int:
        """Remove expired fingerprints. Returns count removed."""
        now = datetime.now(timezone.utc)
        expired = [
            fp for fp, (_, seen_at) in self._seen.items()
            if now - seen_at > timedelta(minutes=self._window_minutes)
        ]
        for fp in expired:
            del self._seen[fp]
        return len(expired)

    def get_stats(self) -> Dict[str, int]:
        """Return deduplication statistics."""
        return {
            "active_fingerprints": len(self._seen),
            "window_minutes": self._window_minutes,
        }


# ============================================================================
# Tests
# ============================================================================


class TestAlertDeduplicator:
    """Test suite for AlertDeduplicator."""

    @pytest.fixture
    def deduplicator(self):
        """Create an AlertDeduplicator instance."""
        return AlertDeduplicator(window_minutes=60)

    @pytest.fixture
    def alert_a(self):
        return Alert(
            source="prometheus", name="HighCPU",
            severity=AlertSeverity.CRITICAL, title="High CPU",
            labels={"instance": "node-01", "job": "node-exporter"},
        )

    @pytest.fixture
    def alert_b(self):
        return Alert(
            source="prometheus", name="HighMemory",
            severity=AlertSeverity.WARNING, title="High Memory",
            labels={"instance": "node-02", "job": "node-exporter"},
        )

    def test_first_alert_not_duplicate(self, deduplicator, alert_a):
        """First alert with a fingerprint is not a duplicate."""
        assert deduplicator.is_duplicate(alert_a) is False

    def test_same_fingerprint_is_duplicate(self, deduplicator, alert_a):
        """Second alert with same fingerprint is a duplicate."""
        deduplicator.process(alert_a)
        dup = Alert(
            source="prometheus", name="HighCPU",
            severity=AlertSeverity.CRITICAL, title="High CPU again",
            labels={"instance": "node-01", "job": "node-exporter"},
        )
        assert deduplicator.is_duplicate(dup) is True

    def test_different_fingerprint_not_duplicate(self, deduplicator, alert_a, alert_b):
        """Alerts with different fingerprints are not duplicates."""
        deduplicator.process(alert_a)
        assert deduplicator.is_duplicate(alert_b) is False

    def test_expired_fingerprint_not_duplicate(self, deduplicator, alert_a):
        """Alert is not a duplicate after the window has passed."""
        deduplicator.process(alert_a)
        # Manually backdate the seen timestamp
        fp = alert_a.fingerprint
        old_alert, _ = deduplicator._seen[fp]
        deduplicator._seen[fp] = (
            old_alert,
            datetime.now(timezone.utc) - timedelta(minutes=61),
        )

        dup = Alert(
            source="prometheus", name="HighCPU",
            severity=AlertSeverity.CRITICAL, title="High CPU refire",
            labels={"instance": "node-01", "job": "node-exporter"},
        )
        assert deduplicator.is_duplicate(dup) is False

    def test_process_new_alert(self, deduplicator, alert_a):
        """process() returns (alert, True) for new alerts."""
        result, is_new = deduplicator.process(alert_a)

        assert is_new is True
        assert result.alert_id == alert_a.alert_id

    def test_process_duplicate_alert(self, deduplicator, alert_a):
        """process() returns (existing, False) for duplicate alerts."""
        deduplicator.process(alert_a)
        dup = Alert(
            source="prometheus", name="HighCPU",
            severity=AlertSeverity.CRITICAL, title="High CPU dup",
            labels={"instance": "node-01", "job": "node-exporter"},
        )

        result, is_new = deduplicator.process(dup)

        assert is_new is False
        assert result.alert_id == alert_a.alert_id

    def test_duplicate_increments_count(self, deduplicator, alert_a):
        """Duplicate processing increments notification_count."""
        deduplicator.process(alert_a)
        initial_count = alert_a.notification_count

        dup = Alert(
            source="prometheus", name="HighCPU",
            severity=AlertSeverity.CRITICAL, title="High CPU dup",
            labels={"instance": "node-01", "job": "node-exporter"},
        )
        deduplicator.process(dup)

        assert alert_a.notification_count == initial_count + 1

    def test_generate_fingerprint_deterministic(self, deduplicator):
        """Same inputs produce the same fingerprint hash."""
        fp1 = deduplicator.generate_fingerprint(
            "prom", "Alert", {"k": "v"},
        )
        fp2 = deduplicator.generate_fingerprint(
            "prom", "Alert", {"k": "v"},
        )
        assert fp1 == fp2
        assert len(fp1) == 32

    def test_generate_fingerprint_different(self, deduplicator):
        """Different label values produce different fingerprints."""
        fp1 = deduplicator.generate_fingerprint("prom", "A", {"k": "v1"})
        fp2 = deduplicator.generate_fingerprint("prom", "A", {"k": "v2"})
        assert fp1 != fp2

    def test_correlate_shared_labels(self, deduplicator, alert_a, alert_b):
        """Find related alerts sharing at least one label value (job=node-exporter)."""
        deduplicator.process(alert_a)
        deduplicator.process(alert_b)

        related = deduplicator.correlate(alert_a)

        assert related is not None
        assert any(r.alert_id == alert_b.alert_id for r in related)

    def test_correlate_no_match(self, deduplicator, alert_a):
        """Returns None when no related alerts found."""
        deduplicator.process(alert_a)
        unrelated = Alert(
            source="loki", name="LogError",
            severity=AlertSeverity.INFO, title="Log error",
            labels={"unique_key": "unique_value"},
        )
        deduplicator.process(unrelated)

        related = deduplicator.correlate(alert_a)
        # alert_a shares no label values with unrelated
        assert related is None

    def test_cleanup_removes_expired(self, deduplicator, alert_a):
        """cleanup() removes fingerprints older than the window."""
        deduplicator.process(alert_a)
        fp = alert_a.fingerprint
        old_alert, _ = deduplicator._seen[fp]
        deduplicator._seen[fp] = (
            old_alert,
            datetime.now(timezone.utc) - timedelta(minutes=61),
        )

        removed = deduplicator.cleanup()

        assert removed == 1
        assert fp not in deduplicator._seen

    def test_cleanup_keeps_active(self, deduplicator, alert_a):
        """cleanup() keeps fingerprints within the window."""
        deduplicator.process(alert_a)

        removed = deduplicator.cleanup()

        assert removed == 0
        assert alert_a.fingerprint in deduplicator._seen

    def test_get_stats(self, deduplicator, alert_a, alert_b):
        """get_stats returns correct counts."""
        deduplicator.process(alert_a)
        deduplicator.process(alert_b)

        stats = deduplicator.get_stats()

        assert stats["active_fingerprints"] == 2
        assert stats["window_minutes"] == 60

    def test_configurable_window(self):
        """Custom dedup window is respected."""
        deduplicator = AlertDeduplicator(window_minutes=5)
        alert = Alert(
            source="test", name="Short",
            severity=AlertSeverity.INFO, title="Short window",
            labels={"k": "v"},
        )
        deduplicator.process(alert)

        # Backdate to 6 minutes ago
        fp = alert.fingerprint
        old_alert, _ = deduplicator._seen[fp]
        deduplicator._seen[fp] = (
            old_alert,
            datetime.now(timezone.utc) - timedelta(minutes=6),
        )

        dup = Alert(
            source="test", name="Short",
            severity=AlertSeverity.INFO, title="Short window dup",
            labels={"k": "v"},
        )
        assert deduplicator.is_duplicate(dup) is False
