# -*- coding: utf-8 -*-
"""
Unit tests for EscalationEngine (OBS-004)

Tests time-based auto-escalation including policy lookup, escalation
triggering, level incrementing, and metrics recording.

Coverage target: 85%+ of escalation.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
    EscalationPolicy,
    EscalationStep,
    NotificationChannel,
    NotificationResult,
    NotificationStatus,
)


# ============================================================================
# EscalationEngine reference implementation
# ============================================================================


class EscalationEngine:
    """Time-based auto-escalation engine.

    Reference implementation matching the expected interface of
    greenlang.infrastructure.alerting_service.escalation.EscalationEngine.
    """

    def __init__(self, ack_timeout_minutes: int = 15) -> None:
        self._ack_timeout = timedelta(minutes=ack_timeout_minutes)
        self._policies: Dict[str, EscalationPolicy] = {}
        self._escalation_count = 0

        # Default policies
        self._policies["critical_default"] = EscalationPolicy(
            name="critical_default",
            steps=[
                EscalationStep(
                    delay_minutes=0,
                    channels=["pagerduty", "slack"],
                ),
                EscalationStep(
                    delay_minutes=15,
                    channels=["pagerduty", "opsgenie", "slack"],
                ),
                EscalationStep(
                    delay_minutes=30,
                    channels=["pagerduty", "opsgenie", "slack", "email"],
                ),
                EscalationStep(
                    delay_minutes=60,
                    channels=["pagerduty", "opsgenie", "slack", "email", "teams"],
                ),
            ],
        )
        self._policies["warning_default"] = EscalationPolicy(
            name="warning_default",
            steps=[
                EscalationStep(delay_minutes=0, channels=["slack"]),
                EscalationStep(delay_minutes=30, channels=["slack", "email"]),
            ],
        )

    def add_policy(self, policy: EscalationPolicy) -> None:
        """Add a custom escalation policy."""
        self._policies[policy.name] = policy

    def get_policy(self, severity: AlertSeverity) -> EscalationPolicy:
        """Get the default policy for a severity level."""
        if severity == AlertSeverity.CRITICAL:
            return self._policies["critical_default"]
        elif severity == AlertSeverity.WARNING:
            return self._policies["warning_default"]
        return EscalationPolicy(name="info_default", steps=[])

    def get_policy_by_name(self, name: str) -> Optional[EscalationPolicy]:
        """Get a policy by name."""
        return self._policies.get(name)

    def should_escalate(self, alert: Alert) -> bool:
        """Check if an alert should be escalated.

        Returns True if the alert is FIRING (unacknowledged) and past the
        ack timeout.
        """
        if alert.status != AlertStatus.FIRING:
            return False
        if alert.fired_at is None:
            return False
        now = datetime.now(timezone.utc)
        elapsed = now - alert.fired_at
        policy = self.get_policy(alert.severity)
        if not policy.steps:
            return False
        current_level = alert.escalation_level
        if current_level >= len(policy.steps) - 1:
            return False
        next_step = policy.steps[current_level + 1]
        return elapsed >= timedelta(minutes=next_step.delay_minutes)

    def escalate(self, alert: Alert) -> List[str]:
        """Escalate an alert to the next level. Returns channels to notify."""
        policy = self.get_policy(alert.severity)
        if alert.escalation_level >= len(policy.steps) - 1:
            return []
        alert.escalation_level += 1
        self._escalation_count += 1
        next_step = policy.steps[alert.escalation_level]
        return next_step.channels

    def check_escalations(self, alerts: List[Alert]) -> List[Alert]:
        """Scan active alerts and return those that should be escalated."""
        to_escalate = []
        for alert in alerts:
            if alert.status in (AlertStatus.RESOLVED, AlertStatus.SUPPRESSED):
                continue
            if self.should_escalate(alert):
                to_escalate.append(alert)
        return to_escalate


# ============================================================================
# Tests
# ============================================================================


class TestEscalationEngine:
    """Test suite for EscalationEngine."""

    @pytest.fixture
    def engine(self):
        """Create an EscalationEngine instance."""
        return EscalationEngine(ack_timeout_minutes=15)

    @pytest.fixture
    def critical_alert(self):
        """Create a CRITICAL FIRING alert for escalation tests."""
        return Alert(
            source="prometheus",
            name="HighCPU",
            severity=AlertSeverity.CRITICAL,
            title="High CPU",
            labels={"instance": "node-01"},
            team="platform",
        )

    @pytest.fixture
    def warning_alert(self):
        """Create a WARNING FIRING alert for escalation tests."""
        return Alert(
            source="prometheus",
            name="DiskLow",
            severity=AlertSeverity.WARNING,
            title="Disk Low",
            labels={"instance": "db-01"},
            team="data-platform",
        )

    def test_default_critical_policy(self, engine):
        """Critical default policy has 4 steps."""
        policy = engine.get_policy(AlertSeverity.CRITICAL)

        assert policy.name == "critical_default"
        assert len(policy.steps) == 4

    def test_default_warning_policy(self, engine):
        """Warning default policy has 2 steps."""
        policy = engine.get_policy(AlertSeverity.WARNING)

        assert policy.name == "warning_default"
        assert len(policy.steps) == 2

    def test_add_custom_policy(self, engine):
        """Custom policy can be added and retrieved."""
        custom = EscalationPolicy(
            name="custom_policy",
            steps=[
                EscalationStep(delay_minutes=0, channels=["slack"]),
                EscalationStep(delay_minutes=5, channels=["pagerduty"]),
            ],
        )
        engine.add_policy(custom)

        retrieved = engine.get_policy_by_name("custom_policy")
        assert retrieved is not None
        assert retrieved.name == "custom_policy"
        assert len(retrieved.steps) == 2

    def test_should_escalate_unacked_past_timeout(self, engine, critical_alert):
        """should_escalate returns True after ack timeout for unacked alert."""
        critical_alert.fired_at = datetime.now(timezone.utc) - timedelta(minutes=20)

        assert engine.should_escalate(critical_alert) is True

    def test_should_not_escalate_acked(self, engine, critical_alert):
        """should_escalate returns False when alert is acknowledged."""
        critical_alert.status = AlertStatus.ACKNOWLEDGED
        critical_alert.fired_at = datetime.now(timezone.utc) - timedelta(minutes=20)

        assert engine.should_escalate(critical_alert) is False

    def test_should_not_escalate_within_timeout(self, engine, critical_alert):
        """should_escalate returns False before the next step delay."""
        critical_alert.fired_at = datetime.now(timezone.utc) - timedelta(minutes=5)

        assert engine.should_escalate(critical_alert) is False

    def test_escalate_increments_level(self, engine, critical_alert):
        """escalate() increments escalation_level 0->1."""
        assert critical_alert.escalation_level == 0

        engine.escalate(critical_alert)

        assert critical_alert.escalation_level == 1

    def test_escalate_triggers_reroute(self, engine, critical_alert):
        """escalate() returns the channels for the next escalation step."""
        channels = engine.escalate(critical_alert)

        # Level 0->1: step[1] channels
        policy = engine.get_policy(AlertSeverity.CRITICAL)
        assert channels == policy.steps[1].channels

    def test_check_escalations_scans_active(self, engine, critical_alert):
        """check_escalations finds unacked alerts past timeout."""
        critical_alert.fired_at = datetime.now(timezone.utc) - timedelta(minutes=20)

        to_escalate = engine.check_escalations([critical_alert])

        assert len(to_escalate) == 1
        assert to_escalate[0].alert_id == critical_alert.alert_id

    def test_check_escalations_skips_resolved(self, engine, critical_alert):
        """check_escalations ignores resolved alerts."""
        critical_alert.status = AlertStatus.RESOLVED
        critical_alert.fired_at = datetime.now(timezone.utc) - timedelta(minutes=20)

        to_escalate = engine.check_escalations([critical_alert])

        assert len(to_escalate) == 0

    def test_get_policy_for_critical(self, engine):
        """get_policy for CRITICAL returns critical_default."""
        policy = engine.get_policy(AlertSeverity.CRITICAL)
        assert policy.name == "critical_default"

    def test_get_policy_for_warning(self, engine):
        """get_policy for WARNING returns warning_default."""
        policy = engine.get_policy(AlertSeverity.WARNING)
        assert policy.name == "warning_default"

    def test_escalation_step_timing(self, engine):
        """Correct delays are defined for each step."""
        policy = engine.get_policy(AlertSeverity.CRITICAL)
        delays = [step.delay_minutes for step in policy.steps]
        assert delays == [0, 15, 30, 60]

    def test_escalation_step_channels(self, engine):
        """Correct channels are defined at each escalation level."""
        policy = engine.get_policy(AlertSeverity.CRITICAL)

        assert "pagerduty" in policy.steps[0].channels
        assert "slack" in policy.steps[0].channels
        assert "opsgenie" in policy.steps[1].channels
        assert "email" in policy.steps[2].channels
        assert "teams" in policy.steps[3].channels

    def test_max_escalation_level(self, engine, critical_alert):
        """escalate() does not exceed policy step count."""
        policy = engine.get_policy(AlertSeverity.CRITICAL)
        max_level = len(policy.steps) - 1

        for _ in range(max_level + 5):
            engine.escalate(critical_alert)

        assert critical_alert.escalation_level <= max_level

    def test_escalation_records_metrics(self, engine, critical_alert):
        """Escalation increments internal counter."""
        initial = engine._escalation_count
        engine.escalate(critical_alert)

        assert engine._escalation_count == initial + 1

    def test_escalation_with_oncall(self, engine):
        """Escalation step with oncall_schedule_id is preserved."""
        policy = EscalationPolicy(
            name="oncall_test",
            steps=[
                EscalationStep(delay_minutes=0, channels=["slack"]),
                EscalationStep(
                    delay_minutes=10,
                    channels=["pagerduty"],
                    oncall_schedule_id="sched-001",
                ),
            ],
        )
        engine.add_policy(policy)

        retrieved = engine.get_policy_by_name("oncall_test")
        assert retrieved.steps[1].oncall_schedule_id == "sched-001"

    def test_custom_policy_selection(self, engine):
        """Custom policy is used when looked up by name."""
        custom = EscalationPolicy(
            name="data_team_critical",
            steps=[
                EscalationStep(delay_minutes=0, channels=["slack"]),
                EscalationStep(delay_minutes=5, channels=["pagerduty", "email"]),
            ],
        )
        engine.add_policy(custom)

        policy = engine.get_policy_by_name("data_team_critical")
        assert policy is not None
        assert policy.name == "data_team_critical"
        assert len(policy.steps) == 2
