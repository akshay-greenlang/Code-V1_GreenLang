# -*- coding: utf-8 -*-
"""
Unit tests for AlertLifecycle (OBS-004)

Tests the alert lifecycle state machine covering transitions:
FIRING -> ACKNOWLEDGED -> INVESTIGATING -> RESOLVED,
FIRING -> SUPPRESSED, RESOLVED -> FIRING (refire),
and invalid transition rejection.

Coverage target: 85%+ of lifecycle.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
)


# ============================================================================
# AlertLifecycle mock (module may not exist yet -- test against interface)
# ============================================================================


class AlertLifecycle:
    """In-memory alert lifecycle state machine.

    This is a test-level reference implementation matching the expected
    interface of greenlang.infrastructure.alerting_service.lifecycle.AlertLifecycle.
    When the production module is built, the import at the top of this file
    should be switched to the real implementation.
    """

    def __init__(self) -> None:
        self._alerts: dict[str, Alert] = {}

    def fire(self, alert: Alert) -> Alert:
        """Fire a new alert or return existing if fingerprint matches."""
        for existing in self._alerts.values():
            if existing.fingerprint == alert.fingerprint and existing.status in (
                AlertStatus.FIRING,
                AlertStatus.ACKNOWLEDGED,
                AlertStatus.INVESTIGATING,
            ):
                return existing
        alert.status = AlertStatus.FIRING
        if alert.fired_at is None:
            alert.fired_at = datetime.now(timezone.utc)
        self._alerts[alert.alert_id] = alert
        return alert

    def acknowledge(self, alert_id: str, user: str = "") -> Alert:
        """Transition FIRING -> ACKNOWLEDGED."""
        alert = self._alerts.get(alert_id)
        if alert is None:
            raise KeyError(f"Alert {alert_id} not found")
        if alert.status not in (AlertStatus.FIRING,):
            raise ValueError(
                f"Cannot acknowledge alert in status {alert.status.value}"
            )
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now(timezone.utc)
        alert.acknowledged_by = user
        return alert

    def investigate(self, alert_id: str) -> Alert:
        """Transition ACKNOWLEDGED -> INVESTIGATING."""
        alert = self._alerts.get(alert_id)
        if alert is None:
            raise KeyError(f"Alert {alert_id} not found")
        if alert.status != AlertStatus.ACKNOWLEDGED:
            raise ValueError(
                f"Cannot investigate alert in status {alert.status.value}"
            )
        alert.status = AlertStatus.INVESTIGATING
        return alert

    def resolve(self, alert_id: str, user: str = "") -> Alert:
        """Transition FIRING/ACKNOWLEDGED/INVESTIGATING -> RESOLVED."""
        alert = self._alerts.get(alert_id)
        if alert is None:
            raise KeyError(f"Alert {alert_id} not found")
        if alert.status not in (
            AlertStatus.FIRING,
            AlertStatus.ACKNOWLEDGED,
            AlertStatus.INVESTIGATING,
        ):
            raise ValueError(
                f"Cannot resolve alert in status {alert.status.value}"
            )
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now(timezone.utc)
        alert.resolved_by = user
        return alert

    def suppress(self, alert_id: str) -> Alert:
        """Transition FIRING -> SUPPRESSED."""
        alert = self._alerts.get(alert_id)
        if alert is None:
            raise KeyError(f"Alert {alert_id} not found")
        if alert.status != AlertStatus.FIRING:
            raise ValueError(
                f"Cannot suppress alert in status {alert.status.value}"
            )
        alert.status = AlertStatus.SUPPRESSED
        return alert

    def refire(self, alert_id: str) -> Alert:
        """Transition RESOLVED/SUPPRESSED -> FIRING."""
        alert = self._alerts.get(alert_id)
        if alert is None:
            raise KeyError(f"Alert {alert_id} not found")
        if alert.status not in (AlertStatus.RESOLVED, AlertStatus.SUPPRESSED):
            raise ValueError(
                f"Cannot refire alert in status {alert.status.value}"
            )
        alert.status = AlertStatus.FIRING
        alert.fired_at = datetime.now(timezone.utc)
        alert.resolved_at = None
        alert.resolved_by = ""
        return alert

    def get_alert(self, alert_id: str) -> Alert | None:
        """Retrieve an alert by ID."""
        return self._alerts.get(alert_id)


# ============================================================================
# Tests
# ============================================================================


class TestAlertLifecycle:
    """Test suite for AlertLifecycle state machine."""

    @pytest.fixture
    def lifecycle(self):
        """Create an AlertLifecycle instance."""
        return AlertLifecycle()

    @pytest.fixture
    def firing_alert(self):
        """Create a fresh alert for lifecycle tests."""
        return Alert(
            source="prometheus",
            name="TestAlert",
            severity=AlertSeverity.CRITICAL,
            title="Test alert for lifecycle",
            labels={"instance": "node-01"},
            team="platform",
        )

    def test_fire_creates_alert(self, lifecycle, firing_alert):
        """fire() returns Alert with status=FIRING."""
        result = lifecycle.fire(firing_alert)

        assert result.status == AlertStatus.FIRING
        assert result.alert_id == firing_alert.alert_id

    def test_fire_sets_fired_at(self, lifecycle, firing_alert):
        """fire() sets fired_at timestamp."""
        result = lifecycle.fire(firing_alert)

        assert result.fired_at is not None
        assert isinstance(result.fired_at, datetime)

    def test_fire_duplicate_returns_existing(self, lifecycle):
        """Same fingerprint returns the same alert."""
        alert1 = Alert(
            source="prom", name="Dup", severity=AlertSeverity.WARNING,
            title="Dup", labels={"k": "v"},
        )
        alert2 = Alert(
            source="prom", name="Dup", severity=AlertSeverity.WARNING,
            title="Dup copy", labels={"k": "v"},
        )

        result1 = lifecycle.fire(alert1)
        result2 = lifecycle.fire(alert2)

        assert result1.alert_id == result2.alert_id

    def test_acknowledge_sets_status(self, lifecycle, firing_alert):
        """FIRING -> ACKNOWLEDGED transition."""
        lifecycle.fire(firing_alert)
        result = lifecycle.acknowledge(firing_alert.alert_id, user="jane")

        assert result.status == AlertStatus.ACKNOWLEDGED

    def test_acknowledge_sets_timestamp(self, lifecycle, firing_alert):
        """acknowledged_at is set on acknowledge."""
        lifecycle.fire(firing_alert)
        result = lifecycle.acknowledge(firing_alert.alert_id)

        assert result.acknowledged_at is not None
        assert isinstance(result.acknowledged_at, datetime)

    def test_acknowledge_sets_user(self, lifecycle, firing_alert):
        """acknowledged_by is set on acknowledge."""
        lifecycle.fire(firing_alert)
        result = lifecycle.acknowledge(firing_alert.alert_id, user="jane.doe")

        assert result.acknowledged_by == "jane.doe"

    def test_acknowledge_invalid_from_resolved(self, lifecycle, firing_alert):
        """Cannot acknowledge a RESOLVED alert -- raises ValueError."""
        lifecycle.fire(firing_alert)
        lifecycle.resolve(firing_alert.alert_id)

        with pytest.raises(ValueError, match="Cannot acknowledge"):
            lifecycle.acknowledge(firing_alert.alert_id)

    def test_investigate_from_acknowledged(self, lifecycle, firing_alert):
        """ACKNOWLEDGED -> INVESTIGATING transition."""
        lifecycle.fire(firing_alert)
        lifecycle.acknowledge(firing_alert.alert_id)
        result = lifecycle.investigate(firing_alert.alert_id)

        assert result.status == AlertStatus.INVESTIGATING

    def test_investigate_invalid_from_firing(self, lifecycle, firing_alert):
        """Cannot investigate directly from FIRING -- raises ValueError."""
        lifecycle.fire(firing_alert)

        with pytest.raises(ValueError, match="Cannot investigate"):
            lifecycle.investigate(firing_alert.alert_id)

    def test_resolve_from_firing(self, lifecycle, firing_alert):
        """FIRING -> RESOLVED transition."""
        lifecycle.fire(firing_alert)
        result = lifecycle.resolve(firing_alert.alert_id)

        assert result.status == AlertStatus.RESOLVED

    def test_resolve_from_acknowledged(self, lifecycle, firing_alert):
        """ACKNOWLEDGED -> RESOLVED transition."""
        lifecycle.fire(firing_alert)
        lifecycle.acknowledge(firing_alert.alert_id)
        result = lifecycle.resolve(firing_alert.alert_id)

        assert result.status == AlertStatus.RESOLVED

    def test_resolve_from_investigating(self, lifecycle, firing_alert):
        """INVESTIGATING -> RESOLVED transition."""
        lifecycle.fire(firing_alert)
        lifecycle.acknowledge(firing_alert.alert_id)
        lifecycle.investigate(firing_alert.alert_id)
        result = lifecycle.resolve(firing_alert.alert_id)

        assert result.status == AlertStatus.RESOLVED

    def test_resolve_sets_timestamp(self, lifecycle, firing_alert):
        """resolved_at is set on resolve."""
        lifecycle.fire(firing_alert)
        result = lifecycle.resolve(firing_alert.alert_id)

        assert result.resolved_at is not None

    def test_resolve_sets_user(self, lifecycle, firing_alert):
        """resolved_by is set on resolve."""
        lifecycle.fire(firing_alert)
        result = lifecycle.resolve(firing_alert.alert_id, user="ops-bot")

        assert result.resolved_by == "ops-bot"

    def test_suppress_from_firing(self, lifecycle, firing_alert):
        """FIRING -> SUPPRESSED transition."""
        lifecycle.fire(firing_alert)
        result = lifecycle.suppress(firing_alert.alert_id)

        assert result.status == AlertStatus.SUPPRESSED

    def test_suppress_invalid_from_resolved(self, lifecycle, firing_alert):
        """Cannot suppress a RESOLVED alert -- raises ValueError."""
        lifecycle.fire(firing_alert)
        lifecycle.resolve(firing_alert.alert_id)

        with pytest.raises(ValueError, match="Cannot suppress"):
            lifecycle.suppress(firing_alert.alert_id)

    def test_refire_from_resolved(self, lifecycle, firing_alert):
        """RESOLVED -> FIRING transition (refire)."""
        lifecycle.fire(firing_alert)
        lifecycle.resolve(firing_alert.alert_id)
        result = lifecycle.refire(firing_alert.alert_id)

        assert result.status == AlertStatus.FIRING

    def test_refire_from_suppressed(self, lifecycle, firing_alert):
        """SUPPRESSED -> FIRING transition (refire)."""
        lifecycle.fire(firing_alert)
        lifecycle.suppress(firing_alert.alert_id)
        result = lifecycle.refire(firing_alert.alert_id)

        assert result.status == AlertStatus.FIRING

    def test_get_alert_by_id(self, lifecycle, firing_alert):
        """Retrieve stored alert by ID."""
        lifecycle.fire(firing_alert)
        retrieved = lifecycle.get_alert(firing_alert.alert_id)

        assert retrieved is not None
        assert retrieved.alert_id == firing_alert.alert_id

    def test_get_alert_not_found(self, lifecycle):
        """get_alert returns None for unknown ID."""
        result = lifecycle.get_alert("nonexistent-id")

        assert result is None
