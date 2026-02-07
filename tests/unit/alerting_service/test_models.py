# -*- coding: utf-8 -*-
"""
Unit tests for Alerting Service Models (OBS-004)

Tests Alert, AlertSeverity, AlertStatus, NotificationResult,
EscalationPolicy, and OnCallUser data models including serialization,
fingerprinting, and defaults.

Coverage target: 85%+ of models.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from unittest.mock import patch

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
    OnCallSchedule,
    OnCallUser,
)


# ============================================================================
# AlertSeverity tests
# ============================================================================


class TestAlertSeverity:
    """Test suite for AlertSeverity enum."""

    def test_values(self):
        """Verify CRITICAL, WARNING, INFO values exist."""
        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.INFO.value == "info"

    def test_string_conversion(self):
        """Verify enum values can be used as strings."""
        assert str(AlertSeverity.CRITICAL) == "AlertSeverity.CRITICAL"
        assert AlertSeverity.CRITICAL == "critical"
        assert AlertSeverity("critical") is AlertSeverity.CRITICAL

    def test_enum_members_count(self):
        """Verify exactly 3 severity levels exist."""
        assert len(AlertSeverity) == 3


# ============================================================================
# AlertStatus tests
# ============================================================================


class TestAlertStatus:
    """Test suite for AlertStatus enum."""

    def test_values(self):
        """Verify FIRING, ACKNOWLEDGED, INVESTIGATING, RESOLVED, SUPPRESSED."""
        assert AlertStatus.FIRING.value == "firing"
        assert AlertStatus.ACKNOWLEDGED.value == "acknowledged"
        assert AlertStatus.INVESTIGATING.value == "investigating"
        assert AlertStatus.RESOLVED.value == "resolved"
        assert AlertStatus.SUPPRESSED.value == "suppressed"

    def test_enum_members_count(self):
        """Verify exactly 5 status values exist."""
        assert len(AlertStatus) == 5


# ============================================================================
# NotificationChannel tests
# ============================================================================


class TestNotificationChannel:
    """Test suite for NotificationChannel enum."""

    def test_all_channels_defined(self):
        """Verify all 6 channels are defined."""
        assert NotificationChannel.PAGERDUTY.value == "pagerduty"
        assert NotificationChannel.OPSGENIE.value == "opsgenie"
        assert NotificationChannel.SLACK.value == "slack"
        assert NotificationChannel.EMAIL.value == "email"
        assert NotificationChannel.TEAMS.value == "teams"
        assert NotificationChannel.WEBHOOK.value == "webhook"
        assert len(NotificationChannel) == 6


# ============================================================================
# Alert tests
# ============================================================================


class TestAlert:
    """Test suite for Alert dataclass."""

    def test_create_alert(self, sample_alert):
        """Test all fields are populated on a sample alert."""
        assert sample_alert.source == "prometheus"
        assert sample_alert.name == "HighCPUUsage"
        assert sample_alert.severity == AlertSeverity.CRITICAL
        assert sample_alert.title == "CPU usage above 95% on node-01"
        assert sample_alert.description != ""
        assert sample_alert.labels["instance"] == "node-01:9090"
        assert sample_alert.tenant_id == "t-acme"
        assert sample_alert.team == "platform"
        assert sample_alert.service == "api-service"
        assert sample_alert.environment == "production"
        assert sample_alert.alert_id != ""
        assert sample_alert.fingerprint != ""

    def test_alert_defaults(self):
        """Test default status=FIRING, escalation_level=0."""
        alert = Alert(
            source="test",
            name="TestAlert",
            severity=AlertSeverity.INFO,
            title="Test",
        )

        assert alert.status == AlertStatus.FIRING
        assert alert.escalation_level == 0
        assert alert.notification_count == 0
        assert alert.tenant_id == ""
        assert alert.acknowledged_by == ""
        assert alert.resolved_by == ""

    def test_generate_fingerprint(self):
        """Test deterministic hash from source+name+labels."""
        fp = Alert.generate_fingerprint(
            source="prometheus",
            name="HighCPU",
            labels={"instance": "node-01", "job": "node-exporter"},
        )

        expected_raw = "prometheus|HighCPU|instance=node-01&job=node-exporter"
        expected_hash = hashlib.md5(expected_raw.encode("utf-8")).hexdigest()

        assert fp == expected_hash
        assert len(fp) == 32  # MD5 hex digest length

    def test_fingerprint_consistency(self):
        """Test same inputs produce the same fingerprint."""
        labels = {"instance": "node-01", "job": "node-exporter"}

        fp1 = Alert.generate_fingerprint("prometheus", "HighCPU", labels)
        fp2 = Alert.generate_fingerprint("prometheus", "HighCPU", labels)

        assert fp1 == fp2

    def test_fingerprint_difference(self):
        """Test different labels produce different fingerprints."""
        fp1 = Alert.generate_fingerprint(
            "prometheus", "HighCPU", {"instance": "node-01"},
        )
        fp2 = Alert.generate_fingerprint(
            "prometheus", "HighCPU", {"instance": "node-02"},
        )

        assert fp1 != fp2

    def test_fingerprint_label_order_independence(self):
        """Test fingerprint is consistent regardless of label insertion order."""
        fp1 = Alert.generate_fingerprint(
            "src", "name", {"b": "2", "a": "1"},
        )
        fp2 = Alert.generate_fingerprint(
            "src", "name", {"a": "1", "b": "2"},
        )

        assert fp1 == fp2

    def test_to_dict(self, sample_alert):
        """Test serialization to dictionary."""
        d = sample_alert.to_dict()

        assert d["source"] == "prometheus"
        assert d["name"] == "HighCPUUsage"
        assert d["severity"] == "critical"
        assert d["status"] == "firing"
        assert d["title"] == "CPU usage above 95% on node-01"
        assert isinstance(d["labels"], dict)
        assert isinstance(d["annotations"], dict)
        assert d["alert_id"] == sample_alert.alert_id
        assert d["fingerprint"] == sample_alert.fingerprint
        assert d["tenant_id"] == "t-acme"
        assert d["team"] == "platform"
        assert d["escalation_level"] == 0
        assert d["fired_at"] is not None

    def test_from_dict(self, sample_alert):
        """Test deserialization round-trip."""
        d = sample_alert.to_dict()
        restored = Alert.from_dict(d)

        assert restored.source == sample_alert.source
        assert restored.name == sample_alert.name
        assert restored.severity == sample_alert.severity
        assert restored.status == sample_alert.status
        assert restored.title == sample_alert.title
        assert restored.labels == sample_alert.labels
        assert restored.fingerprint == sample_alert.fingerprint
        assert restored.tenant_id == sample_alert.tenant_id
        assert restored.escalation_level == sample_alert.escalation_level

    def test_optional_fields_none(self):
        """Test tenant_id, acknowledged_at etc can be empty/None."""
        alert = Alert(
            source="test",
            name="TestAlert",
            severity=AlertSeverity.INFO,
            title="Test",
        )

        assert alert.tenant_id == ""
        assert alert.acknowledged_at is None
        assert alert.resolved_at is None
        assert alert.acknowledged_by == ""
        assert alert.resolved_by == ""
        assert alert.runbook_url == ""
        assert alert.dashboard_url == ""
        assert alert.related_trace_id == ""

    def test_fired_at_auto_set(self):
        """Test fired_at is automatically set to now if not provided."""
        alert = Alert(
            source="test",
            name="TestAlert",
            severity=AlertSeverity.INFO,
            title="Test",
        )

        assert alert.fired_at is not None
        assert isinstance(alert.fired_at, datetime)
        assert alert.fired_at.tzinfo == timezone.utc

    def test_fingerprint_auto_generated(self):
        """Test fingerprint is auto-generated from source+name+labels."""
        alert = Alert(
            source="test",
            name="TestAlert",
            severity=AlertSeverity.INFO,
            title="Test",
            labels={"key": "val"},
        )

        expected = Alert.generate_fingerprint("test", "TestAlert", {"key": "val"})
        assert alert.fingerprint == expected

    def test_from_dict_with_string_severity(self):
        """Test from_dict handles string severity values."""
        data = {
            "source": "test",
            "name": "TestAlert",
            "severity": "warning",
            "title": "Test",
        }

        alert = Alert.from_dict(data)
        assert alert.severity == AlertSeverity.WARNING

    def test_from_dict_with_string_status(self):
        """Test from_dict handles string status values."""
        data = {
            "source": "test",
            "name": "TestAlert",
            "severity": "info",
            "status": "acknowledged",
            "title": "Test",
        }

        alert = Alert.from_dict(data)
        assert alert.status == AlertStatus.ACKNOWLEDGED

    def test_from_dict_with_iso_timestamps(self):
        """Test from_dict parses ISO timestamp strings."""
        data = {
            "source": "test",
            "name": "TestAlert",
            "severity": "info",
            "title": "Test",
            "fired_at": "2026-02-07T10:00:00+00:00",
            "acknowledged_at": "2026-02-07T10:05:00+00:00",
            "resolved_at": "2026-02-07T11:00:00+00:00",
        }

        alert = Alert.from_dict(data)
        assert isinstance(alert.fired_at, datetime)
        assert isinstance(alert.acknowledged_at, datetime)
        assert isinstance(alert.resolved_at, datetime)


# ============================================================================
# NotificationResult tests
# ============================================================================


class TestNotificationResult:
    """Test suite for NotificationResult dataclass."""

    def test_success_result(self):
        """Test creating a successful notification result."""
        result = NotificationResult(
            channel=NotificationChannel.SLACK,
            status=NotificationStatus.SENT,
            recipient="https://hooks.slack.com/test",
            duration_ms=150.5,
            response_code=200,
        )

        assert result.channel == NotificationChannel.SLACK
        assert result.status == NotificationStatus.SENT
        assert result.duration_ms == 150.5
        assert result.response_code == 200
        assert result.error_message == ""
        assert result.sent_at is not None

    def test_failure_result(self):
        """Test creating a failed notification result."""
        result = NotificationResult(
            channel=NotificationChannel.PAGERDUTY,
            status=NotificationStatus.FAILED,
            error_message="Connection timeout",
            response_code=504,
        )

        assert result.status == NotificationStatus.FAILED
        assert result.error_message == "Connection timeout"
        assert result.response_code == 504

    def test_rate_limited_result(self):
        """Test creating a rate-limited notification result."""
        result = NotificationResult(
            channel=NotificationChannel.OPSGENIE,
            status=NotificationStatus.RATE_LIMITED,
            error_message="Rate limit exceeded",
            response_code=429,
        )

        assert result.status == NotificationStatus.RATE_LIMITED
        assert result.response_code == 429

    def test_sent_at_auto_set(self):
        """Test sent_at is auto-populated."""
        result = NotificationResult(
            channel=NotificationChannel.EMAIL,
            status=NotificationStatus.SENT,
        )

        assert result.sent_at is not None
        assert isinstance(result.sent_at, datetime)


# ============================================================================
# EscalationPolicy tests
# ============================================================================


class TestEscalationPolicy:
    """Test suite for EscalationPolicy and EscalationStep."""

    def test_create_policy_with_steps(self, sample_escalation_policy):
        """Test policy creation with 3 escalation steps."""
        assert sample_escalation_policy.name == "critical_default"
        assert len(sample_escalation_policy.steps) == 3
        assert sample_escalation_policy.steps[0].delay_minutes == 0
        assert sample_escalation_policy.steps[1].delay_minutes == 15
        assert sample_escalation_policy.steps[2].delay_minutes == 30

    def test_step_delay_ordering(self, sample_escalation_policy):
        """Test escalation steps are ordered by delay."""
        delays = [step.delay_minutes for step in sample_escalation_policy.steps]
        assert delays == sorted(delays)

    def test_step_channels(self, sample_escalation_policy):
        """Test each step has the correct channels."""
        assert "pagerduty" in sample_escalation_policy.steps[0].channels
        assert "slack" in sample_escalation_policy.steps[0].channels
        assert "email" in sample_escalation_policy.steps[2].channels

    def test_step_default_repeat(self):
        """Test step default repeat is 1."""
        step = EscalationStep(delay_minutes=10)
        assert step.repeat == 1
        assert step.channels == []
        assert step.oncall_schedule_id == ""
        assert step.notify_users == []

    def test_empty_policy(self):
        """Test policy with no steps."""
        policy = EscalationPolicy(name="empty")
        assert policy.name == "empty"
        assert policy.steps == []


# ============================================================================
# OnCallUser tests
# ============================================================================


class TestOnCallUser:
    """Test suite for OnCallUser dataclass."""

    def test_create_user(self, sample_oncall_user):
        """Test creating an OnCallUser with all fields."""
        assert sample_oncall_user.user_id == "usr-pd-001"
        assert sample_oncall_user.name == "Jane Doe"
        assert sample_oncall_user.email == "jane.doe@greenlang.io"
        assert sample_oncall_user.phone == "+15551234567"
        assert sample_oncall_user.provider == "pagerduty"
        assert sample_oncall_user.schedule_id == "sched-001"

    def test_user_fields(self):
        """Test OnCallUser with minimal fields."""
        user = OnCallUser(user_id="usr-001", name="Test User")

        assert user.user_id == "usr-001"
        assert user.name == "Test User"
        assert user.email == ""
        assert user.phone == ""
        assert user.provider == ""
        assert user.schedule_id == ""


# ============================================================================
# OnCallSchedule tests
# ============================================================================


class TestOnCallSchedule:
    """Test suite for OnCallSchedule dataclass."""

    def test_create_schedule(self, sample_oncall_user):
        """Test creating an OnCallSchedule with current on-call user."""
        schedule = OnCallSchedule(
            schedule_id="sched-001",
            name="Platform On-Call",
            provider="pagerduty",
            current_oncall=sample_oncall_user,
            timezone="US/Eastern",
        )

        assert schedule.schedule_id == "sched-001"
        assert schedule.name == "Platform On-Call"
        assert schedule.provider == "pagerduty"
        assert schedule.current_oncall.name == "Jane Doe"
        assert schedule.timezone == "US/Eastern"

    def test_schedule_defaults(self):
        """Test schedule defaults."""
        schedule = OnCallSchedule(schedule_id="s-001", name="Test")

        assert schedule.provider == ""
        assert schedule.current_oncall is None
        assert schedule.timezone == "UTC"
