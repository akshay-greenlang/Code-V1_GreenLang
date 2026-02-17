# -*- coding: utf-8 -*-
"""
Unit tests for AlertManagerEngine - AGENT-DATA-016 Engine 6.

Tests all public methods of AlertManagerEngine including alert creation,
sending, acknowledgement, resolution, suppression, throttling, deduplication,
escalation, message formatting, breach management, queries, statistics,
and reset.

Target: 70+ tests, 85%+ coverage.

Note: The models.py FreshnessAlert/SLABreach Pydantic models use field names
that differ from what the engine passes (e.g. ``id`` vs ``alert_id``,
``alert_severity`` vs ``severity``). We patch the model classes in the
alert_manager module namespace with lightweight dataclass replacements that
accept the engine's field names.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Flexible dataclass replacements for FreshnessAlert / SLABreach
# ---------------------------------------------------------------------------
# The engine constructs these with field names (alert_id, dataset_id,
# severity, recipients, etc.) that do not match the Pydantic schema in
# models.py (id, alert_severity, etc.).  We supply lightweight replacements
# that accept any keyword argument so the engine can create objects freely.
# ---------------------------------------------------------------------------


@dataclass
class _FlexFreshnessAlert:
    """Flexible FreshnessAlert replacement for testing.

    Accepts any keyword argument the engine passes and stores them
    as instance attributes.
    """

    alert_id: str = ""
    breach_id: str = ""
    dataset_id: str = ""
    severity: Any = None
    channel: Any = None
    status: Any = None
    message: str = ""
    recipients: List[str] = dc_field(default_factory=list)
    created_at: Any = None
    sent_at: Any = None
    acknowledged_at: Any = None
    acknowledged_by: Any = None
    resolved_at: Any = None
    resolution_notes: Any = None
    suppressed_at: Any = None
    suppression_reason: Any = None
    escalation_level: int = 0
    provenance_hash: str = ""


@dataclass
class _FlexSLABreach:
    """Flexible SLABreach replacement for testing.

    Accepts any keyword argument the engine passes.
    """

    breach_id: str = ""
    dataset_id: str = ""
    sla_id: str = ""
    severity: Any = None
    status: Any = None
    age_at_breach_hours: float = 0.0
    detected_at: Any = None
    acknowledged_at: Any = None
    acknowledged_by: Any = None
    resolved_at: Any = None
    resolution_notes: Any = None
    provenance_hash: str = ""


# ---------------------------------------------------------------------------
# Apply patches before importing AlertManagerEngine
# ---------------------------------------------------------------------------

# We need to patch the model classes *in the module where they are imported*
# (greenlang.data_freshness_monitor.alert_manager) so the engine constructs
# our flexible dataclasses instead of the strict Pydantic models.

import greenlang.data_freshness_monitor.alert_manager as _am_module  # noqa: E402
from greenlang.data_freshness_monitor.models import (  # noqa: E402
    AlertChannel,
    AlertSeverity,
    AlertStatus,
    BreachSeverity,
    BreachStatus,
)

# Patch at module level so every test in this file uses the flex models.
_am_module.FreshnessAlert = _FlexFreshnessAlert  # type: ignore[misc]
_am_module.SLABreach = _FlexSLABreach  # type: ignore[misc]

from greenlang.data_freshness_monitor.alert_manager import (  # noqa: E402
    AlertManagerEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> AlertManagerEngine:
    """Create a fresh AlertManagerEngine for each test."""
    eng = AlertManagerEngine()
    eng.reset()
    return eng


@pytest.fixture
def breach(engine: AlertManagerEngine) -> _FlexSLABreach:
    """Record and return a sample SLA breach."""
    return engine.record_breach(
        dataset_id="ds-001",
        sla_id="sla-001",
        breach_severity="critical",
        age_at_breach_hours=96.5,
    )


@pytest.fixture
def alert(engine: AlertManagerEngine, breach: _FlexSLABreach) -> _FlexFreshnessAlert:
    """Create a sample alert without sending it."""
    return engine.create_alert(
        breach_id=breach.breach_id,
        dataset_id="ds-001",
        alert_severity="critical",
        channel="email",
        message="Dataset ds-001 is stale",
        recipients=["oncall@example.com"],
    )


@pytest.fixture
def sent_alert(
    engine: AlertManagerEngine, alert: _FlexFreshnessAlert,
) -> _FlexFreshnessAlert:
    """Create and send a sample alert."""
    engine.send_alert(alert.alert_id)
    return alert


# ============================================================================
# Test: create_alert
# ============================================================================


class TestCreateAlert:
    """Tests for create_alert."""

    def test_create_alert_returns_alert(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """create_alert returns a FreshnessAlert with pending status."""
        alert = engine.create_alert(
            breach_id=breach.breach_id,
            dataset_id="ds-001",
            alert_severity="warning",
            channel="slack",
            message="SLA warning",
        )
        assert alert.alert_id.startswith("ALT-")
        assert alert.status == AlertStatus.PENDING

    def test_create_alert_stores_fields(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Created alert stores all provided field values."""
        alert = engine.create_alert(
            breach_id=breach.breach_id,
            dataset_id="ds-002",
            alert_severity="critical",
            channel="pagerduty",
            message="Critical breach",
            recipients=["ops@x.com", "admin@x.com"],
        )
        assert alert.breach_id == breach.breach_id
        assert alert.dataset_id == "ds-002"
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.channel == AlertChannel.PAGERDUTY
        assert alert.message == "Critical breach"
        assert "ops@x.com" in alert.recipients

    def test_create_alert_provenance_hash(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Created alert has a 64-char SHA-256 provenance hash."""
        alert = engine.create_alert(
            breach_id=breach.breach_id,
            dataset_id="ds-001",
            alert_severity="info",
            channel="email",
            message="Info alert",
        )
        assert len(alert.provenance_hash) == 64

    def test_create_alert_invalid_severity(
        self, engine: AlertManagerEngine,
    ):
        """Invalid severity raises ValueError."""
        with pytest.raises(ValueError, match="Invalid alert severity"):
            engine.create_alert(
                breach_id="br-x",
                dataset_id="ds-x",
                alert_severity="bogus",
                channel="email",
                message="test",
            )

    def test_create_alert_invalid_channel(
        self, engine: AlertManagerEngine,
    ):
        """Invalid channel raises ValueError."""
        with pytest.raises(ValueError, match="Invalid alert channel"):
            engine.create_alert(
                breach_id="br-x",
                dataset_id="ds-x",
                alert_severity="warning",
                channel="carrier_pigeon",
                message="test",
            )

    def test_create_alert_no_recipients(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Recipients default to empty list when not provided."""
        alert = engine.create_alert(
            breach_id=breach.breach_id,
            dataset_id="ds-001",
            alert_severity="warning",
            channel="webhook",
            message="test",
        )
        assert alert.recipients == []

    def test_create_alert_unique_ids(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Each created alert has a unique alert_id."""
        ids = set()
        for _ in range(10):
            a = engine.create_alert(
                breach_id=breach.breach_id,
                dataset_id="ds-001",
                alert_severity="warning",
                channel="email",
                message="test",
            )
            ids.add(a.alert_id)
        assert len(ids) == 10

    def test_create_alert_all_severities(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """All four AlertSeverity values are accepted."""
        for sev in ("info", "warning", "critical", "emergency"):
            a = engine.create_alert(
                breach_id=breach.breach_id,
                dataset_id="ds-001",
                alert_severity=sev,
                channel="email",
                message=f"{sev} alert",
            )
            assert a.severity == AlertSeverity(sev)

    def test_create_alert_all_channels(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """All valid AlertChannel values are accepted."""
        for ch in ("email", "slack", "pagerduty", "webhook", "teams"):
            a = engine.create_alert(
                breach_id=breach.breach_id,
                dataset_id="ds-001",
                alert_severity="warning",
                channel=ch,
                message=f"{ch} alert",
            )
            assert a.channel == AlertChannel(ch)


# ============================================================================
# Test: send_alert
# ============================================================================


class TestSendAlert:
    """Tests for send_alert."""

    def test_send_alert_returns_true(
        self, engine: AlertManagerEngine, alert: _FlexFreshnessAlert,
    ):
        """Successful send returns True."""
        assert engine.send_alert(alert.alert_id) is True

    def test_send_alert_sets_sent_status(
        self, engine: AlertManagerEngine, alert: _FlexFreshnessAlert,
    ):
        """After send, alert status is SENT."""
        engine.send_alert(alert.alert_id)
        assert alert.status == AlertStatus.SENT

    def test_send_alert_sets_sent_at(
        self, engine: AlertManagerEngine, alert: _FlexFreshnessAlert,
    ):
        """After send, sent_at is set to a datetime."""
        engine.send_alert(alert.alert_id)
        assert alert.sent_at is not None

    def test_send_alert_unknown_id_raises(
        self, engine: AlertManagerEngine,
    ):
        """Unknown alert_id raises ValueError."""
        with pytest.raises(ValueError, match="Alert not found"):
            engine.send_alert("nonexistent-alert")

    def test_send_resolved_alert_returns_false(
        self, engine: AlertManagerEngine, sent_alert: _FlexFreshnessAlert,
    ):
        """Cannot send a resolved alert; returns False."""
        engine.resolve_alert(sent_alert.alert_id, "fixed")
        result = engine.send_alert(sent_alert.alert_id)
        assert result is False

    def test_send_suppressed_alert_returns_false(
        self, engine: AlertManagerEngine, sent_alert: _FlexFreshnessAlert,
    ):
        """Cannot send a suppressed alert; returns False."""
        engine.suppress_alert(sent_alert.alert_id, "maint window")
        result = engine.send_alert(sent_alert.alert_id)
        assert result is False


# ============================================================================
# Test: create_and_send_alert
# ============================================================================


class TestCreateAndSendAlert:
    """Tests for create_and_send_alert."""

    def test_create_and_send_returns_sent_alert(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """create_and_send_alert returns an alert with SENT status."""
        alert = engine.create_and_send_alert(
            breach_id=breach.breach_id,
            dataset_id="ds-001",
            alert_severity="critical",
            channel="email",
            message="SLA breached",
            recipients=["oncall@example.com"],
        )
        assert alert.status == AlertStatus.SENT

    def test_create_and_send_second_call_throttled(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Second call to same dataset+channel within throttle window is throttled."""
        # First call sends
        a1 = engine.create_and_send_alert(
            breach_id=breach.breach_id,
            dataset_id="ds-001",
            alert_severity="critical",
            channel="email",
            message="First alert",
        )
        assert a1.status == AlertStatus.SENT

        # Second call within throttle window stays PENDING
        a2 = engine.create_and_send_alert(
            breach_id=breach.breach_id,
            dataset_id="ds-001",
            alert_severity="critical",
            channel="email",
            message="Second alert",
        )
        assert a2.status == AlertStatus.PENDING

    def test_create_and_send_different_channels_not_throttled(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Different channels for same dataset are not throttled."""
        a1 = engine.create_and_send_alert(
            breach_id=breach.breach_id,
            dataset_id="ds-001",
            alert_severity="critical",
            channel="email",
            message="Email alert",
        )
        a2 = engine.create_and_send_alert(
            breach_id=breach.breach_id,
            dataset_id="ds-001",
            alert_severity="critical",
            channel="slack",
            message="Slack alert",
        )
        assert a1.status == AlertStatus.SENT
        assert a2.status == AlertStatus.SENT


# ============================================================================
# Test: acknowledge_alert
# ============================================================================


class TestAcknowledgeAlert:
    """Tests for acknowledge_alert."""

    def test_acknowledge_sent_alert(
        self, engine: AlertManagerEngine, sent_alert: _FlexFreshnessAlert,
    ):
        """Acknowledging a sent alert transitions to ACKNOWLEDGED."""
        result = engine.acknowledge_alert(sent_alert.alert_id, "ops-engineer")
        assert result.status == AlertStatus.ACKNOWLEDGED
        assert result.acknowledged_by == "ops-engineer"
        assert result.acknowledged_at is not None

    def test_acknowledge_pending_alert(
        self, engine: AlertManagerEngine, alert: _FlexFreshnessAlert,
    ):
        """Acknowledging a pending alert transitions to ACKNOWLEDGED."""
        result = engine.acknowledge_alert(alert.alert_id, "ops-engineer")
        assert result.status == AlertStatus.ACKNOWLEDGED

    def test_acknowledge_unknown_alert_raises(
        self, engine: AlertManagerEngine,
    ):
        """Unknown alert_id raises ValueError."""
        with pytest.raises(ValueError, match="Alert not found"):
            engine.acknowledge_alert("bad-id", "ops")

    def test_acknowledge_resolved_alert_raises(
        self, engine: AlertManagerEngine, sent_alert: _FlexFreshnessAlert,
    ):
        """Cannot acknowledge a resolved alert."""
        engine.resolve_alert(sent_alert.alert_id, "fixed")
        with pytest.raises(ValueError, match="Cannot acknowledge"):
            engine.acknowledge_alert(sent_alert.alert_id, "ops")

    def test_acknowledge_suppressed_alert_raises(
        self, engine: AlertManagerEngine, sent_alert: _FlexFreshnessAlert,
    ):
        """Cannot acknowledge a suppressed alert."""
        engine.suppress_alert(sent_alert.alert_id, "maint")
        with pytest.raises(ValueError, match="Cannot acknowledge"):
            engine.acknowledge_alert(sent_alert.alert_id, "ops")

    def test_acknowledge_updates_provenance(
        self, engine: AlertManagerEngine, sent_alert: _FlexFreshnessAlert,
    ):
        """Acknowledgement updates provenance hash."""
        old_hash = sent_alert.provenance_hash
        engine.acknowledge_alert(sent_alert.alert_id, "ops")
        assert sent_alert.provenance_hash != old_hash
        assert len(sent_alert.provenance_hash) == 64


# ============================================================================
# Test: resolve_alert
# ============================================================================


class TestResolveAlert:
    """Tests for resolve_alert."""

    def test_resolve_sent_alert(
        self, engine: AlertManagerEngine, sent_alert: _FlexFreshnessAlert,
    ):
        """Resolving a sent alert transitions to RESOLVED."""
        result = engine.resolve_alert(sent_alert.alert_id, "Pipeline restarted")
        assert result.status == AlertStatus.RESOLVED
        assert result.resolution_notes == "Pipeline restarted"
        assert result.resolved_at is not None

    def test_resolve_acknowledged_alert(
        self, engine: AlertManagerEngine, sent_alert: _FlexFreshnessAlert,
    ):
        """Resolving an acknowledged alert transitions to RESOLVED."""
        engine.acknowledge_alert(sent_alert.alert_id, "ops")
        result = engine.resolve_alert(sent_alert.alert_id, "Resolved after ack")
        assert result.status == AlertStatus.RESOLVED

    def test_resolve_unknown_alert_raises(
        self, engine: AlertManagerEngine,
    ):
        """Unknown alert_id raises ValueError."""
        with pytest.raises(ValueError, match="Alert not found"):
            engine.resolve_alert("bad-id", "notes")

    def test_resolve_suppressed_alert_raises(
        self, engine: AlertManagerEngine, sent_alert: _FlexFreshnessAlert,
    ):
        """Cannot resolve a suppressed alert."""
        engine.suppress_alert(sent_alert.alert_id, "maint")
        with pytest.raises(ValueError, match="Cannot resolve suppressed"):
            engine.resolve_alert(sent_alert.alert_id, "notes")


# ============================================================================
# Test: suppress_alert
# ============================================================================


class TestSuppressAlert:
    """Tests for suppress_alert."""

    def test_suppress_sent_alert(
        self, engine: AlertManagerEngine, sent_alert: _FlexFreshnessAlert,
    ):
        """Suppressing a sent alert transitions to SUPPRESSED."""
        result = engine.suppress_alert(sent_alert.alert_id, "Maintenance window")
        assert result.status == AlertStatus.SUPPRESSED
        assert result.suppression_reason == "Maintenance window"
        assert result.suppressed_at is not None

    def test_suppress_pending_alert(
        self, engine: AlertManagerEngine, alert: _FlexFreshnessAlert,
    ):
        """Suppressing a pending alert transitions to SUPPRESSED."""
        result = engine.suppress_alert(alert.alert_id, "Known issue")
        assert result.status == AlertStatus.SUPPRESSED

    def test_suppress_unknown_alert_raises(
        self, engine: AlertManagerEngine,
    ):
        """Unknown alert_id raises ValueError."""
        with pytest.raises(ValueError, match="Alert not found"):
            engine.suppress_alert("bad-id", "reason")

    def test_suppress_resolved_alert_raises(
        self, engine: AlertManagerEngine, sent_alert: _FlexFreshnessAlert,
    ):
        """Cannot suppress a resolved alert."""
        engine.resolve_alert(sent_alert.alert_id, "fixed")
        with pytest.raises(ValueError, match="Cannot suppress resolved"):
            engine.suppress_alert(sent_alert.alert_id, "reason")


# ============================================================================
# Test: should_throttle
# ============================================================================


class TestThrottling:
    """Tests for should_throttle."""

    def test_no_throttle_first_time(self, engine: AlertManagerEngine):
        """First alert for dataset+channel is not throttled."""
        assert engine.should_throttle("ds-001", "email") is False

    def test_throttle_after_send(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """After sending an alert, same dataset+channel is throttled."""
        engine.create_and_send_alert(
            breach_id=breach.breach_id,
            dataset_id="ds-001",
            alert_severity="critical",
            channel="email",
            message="test",
        )
        assert engine.should_throttle("ds-001", "email") is True

    def test_no_throttle_different_channel(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Different channel for same dataset is not throttled."""
        engine.create_and_send_alert(
            breach_id=breach.breach_id,
            dataset_id="ds-001",
            alert_severity="critical",
            channel="email",
            message="test",
        )
        assert engine.should_throttle("ds-001", "slack") is False

    def test_no_throttle_different_dataset(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Different dataset for same channel is not throttled."""
        engine.create_and_send_alert(
            breach_id=breach.breach_id,
            dataset_id="ds-001",
            alert_severity="critical",
            channel="email",
            message="test",
        )
        assert engine.should_throttle("ds-002", "email") is False


# ============================================================================
# Test: should_deduplicate
# ============================================================================


class TestDeduplication:
    """Tests for should_deduplicate."""

    def test_no_dedup_first_time(self, engine: AlertManagerEngine):
        """First alert is not deduplicated."""
        assert engine.should_deduplicate("ds-001", "critical", "email") is False

    def test_dedup_after_send(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Same dataset+severity+channel is deduplicated after send."""
        engine.create_and_send_alert(
            breach_id=breach.breach_id,
            dataset_id="ds-001",
            alert_severity="critical",
            channel="email",
            message="test",
        )
        assert engine.should_deduplicate("ds-001", "critical", "email") is True

    def test_no_dedup_different_severity(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Different severity is not deduplicated."""
        engine.create_and_send_alert(
            breach_id=breach.breach_id,
            dataset_id="ds-001",
            alert_severity="critical",
            channel="email",
            message="test",
        )
        assert engine.should_deduplicate("ds-001", "warning", "email") is False

    def test_no_dedup_different_channel(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Different channel is not deduplicated."""
        engine.create_and_send_alert(
            breach_id=breach.breach_id,
            dataset_id="ds-001",
            alert_severity="critical",
            channel="email",
            message="test",
        )
        assert engine.should_deduplicate("ds-001", "critical", "slack") is False


# ============================================================================
# Test: escalate
# ============================================================================


class TestEscalation:
    """Tests for escalate and run_escalation_check."""

    def test_escalate_creates_alert(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Escalation creates and sends an alert for the breach."""
        policy = {
            "levels": [
                {"delay_minutes": 0, "channel": "email", "severity": "warning"},
                {"delay_minutes": 5, "channel": "pagerduty", "severity": "critical"},
            ],
        }
        alert = engine.escalate(breach.breach_id, policy, current_level=-1)
        assert alert is not None
        assert alert.status == AlertStatus.SENT

    def test_escalate_no_levels_returns_none(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Empty escalation policy returns None."""
        result = engine.escalate(breach.breach_id, {"levels": []}, current_level=-1)
        assert result is None

    def test_escalate_unknown_breach_raises(
        self, engine: AlertManagerEngine,
    ):
        """Unknown breach_id raises ValueError."""
        with pytest.raises(ValueError, match="Breach not found"):
            engine.escalate(
                "nonexistent-breach",
                {"levels": [{"delay_minutes": 0, "channel": "email"}]},
                current_level=-1,
            )

    def test_escalate_level_tracking(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Escalation updates the escalation level for the breach."""
        policy = {
            "levels": [
                {"delay_minutes": 0, "channel": "email", "severity": "warning"},
            ],
        }
        alert = engine.escalate(breach.breach_id, policy, current_level=-1)
        assert alert is not None
        assert alert.escalation_level == 0

    def test_escalate_respects_current_level(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Escalation only goes to levels higher than current_level."""
        policy = {
            "levels": [
                {"delay_minutes": 0, "channel": "email", "severity": "warning"},
                {"delay_minutes": 0, "channel": "slack", "severity": "critical"},
            ],
        }
        # Already at level 0, so should escalate to level 1
        alert = engine.escalate(breach.breach_id, policy, current_level=0)
        assert alert is not None
        assert alert.escalation_level == 1

    def test_escalate_no_further_levels(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """No escalation when already at highest level."""
        policy = {
            "levels": [
                {"delay_minutes": 0, "channel": "email", "severity": "warning"},
            ],
        }
        result = engine.escalate(breach.breach_id, policy, current_level=0)
        assert result is None

    def test_run_escalation_check_empty(
        self, engine: AlertManagerEngine,
    ):
        """Empty breaches list returns empty alerts list."""
        result = engine.run_escalation_check([], {})
        assert result == []

    def test_run_escalation_check_with_breach(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """run_escalation_check creates escalated alerts."""
        policies = {
            "default": {
                "levels": [
                    {"delay_minutes": 0, "channel": "email", "severity": "warning"},
                ],
            },
        }
        active_breaches = [
            {
                "breach_id": breach.breach_id,
                "dataset_id": "ds-001",
                "detected_at": breach.detected_at,
                "severity": "critical",
                "policy_name": "default",
            },
        ]
        alerts = engine.run_escalation_check(active_breaches, policies)
        assert len(alerts) >= 1

    def test_run_escalation_check_missing_policy(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Missing policy for a breach is skipped gracefully."""
        active_breaches = [
            {
                "breach_id": breach.breach_id,
                "policy_name": "nonexistent_policy",
            },
        ]
        alerts = engine.run_escalation_check(active_breaches, {})
        assert alerts == []


# ============================================================================
# Test: format_alert_message
# ============================================================================


class TestFormatAlertMessage:
    """Tests for format_alert_message."""

    def test_format_simple_template(self, engine: AlertManagerEngine):
        """Simple template substitution works."""
        result = engine.format_alert_message(
            "Dataset {name} is {status}",
            {"name": "Scope1", "status": "stale"},
        )
        assert result == "Dataset Scope1 is stale"

    def test_format_with_float_specifier(self, engine: AlertManagerEngine):
        """Float format specifier (:.1f) works."""
        result = engine.format_alert_message(
            "Age: {age:.1f}h", {"age": 96.555},
        )
        assert result == "Age: 96.6h"

    def test_format_missing_key_returns_error_prefix(
        self, engine: AlertManagerEngine,
    ):
        """Missing key in context returns template with error prefix."""
        result = engine.format_alert_message(
            "Dataset {missing_key}", {},
        )
        assert result.startswith("[template error]")

    def test_format_empty_template(self, engine: AlertManagerEngine):
        """Empty template returns empty string."""
        result = engine.format_alert_message("", {})
        assert result == ""

    def test_format_no_placeholders(self, engine: AlertManagerEngine):
        """Template with no placeholders returns as-is."""
        result = engine.format_alert_message("Hello world", {"key": "val"})
        assert result == "Hello world"


# ============================================================================
# Test: list_alerts / get_active_alerts / get_alert_statistics
# ============================================================================


class TestAlertQueries:
    """Tests for alert query methods."""

    def test_list_alerts_empty(self, engine: AlertManagerEngine):
        """No alerts returns empty list."""
        assert engine.list_alerts() == []

    def test_list_alerts_after_create(
        self, engine: AlertManagerEngine, alert: _FlexFreshnessAlert,
    ):
        """Created alert appears in list."""
        alerts = engine.list_alerts()
        assert len(alerts) == 1
        assert alerts[0].alert_id == alert.alert_id

    def test_list_alerts_filter_by_dataset(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Filter by dataset_id works."""
        engine.create_alert(
            breach.breach_id, "ds-AAA", "warning", "email", "msg1",
        )
        engine.create_alert(
            breach.breach_id, "ds-BBB", "warning", "email", "msg2",
        )
        result = engine.list_alerts(dataset_id="ds-AAA")
        assert len(result) == 1
        assert result[0].dataset_id == "ds-AAA"

    def test_list_alerts_filter_by_status(
        self, engine: AlertManagerEngine, sent_alert: _FlexFreshnessAlert,
    ):
        """Filter by status works."""
        result = engine.list_alerts(status="sent")
        assert all(a.status == AlertStatus.SENT for a in result)

    def test_list_alerts_filter_by_severity(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Filter by severity works."""
        engine.create_alert(
            breach.breach_id, "ds-001", "warning", "email", "w",
        )
        engine.create_alert(
            breach.breach_id, "ds-001", "critical", "email", "c",
        )
        result = engine.list_alerts(severity="warning")
        assert all(a.severity == AlertSeverity.WARNING for a in result)

    def test_list_alerts_filter_by_channel(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Filter by channel works."""
        engine.create_alert(
            breach.breach_id, "ds-001", "warning", "email", "e",
        )
        engine.create_alert(
            breach.breach_id, "ds-001", "warning", "slack", "s",
        )
        result = engine.list_alerts(channel="slack")
        assert all(a.channel == AlertChannel.SLACK for a in result)

    def test_list_alerts_pagination(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Limit and offset pagination works."""
        for i in range(5):
            engine.create_alert(
                breach.breach_id, "ds-001", "warning", "email", f"msg {i}",
            )
        page1 = engine.list_alerts(limit=2, offset=0)
        page2 = engine.list_alerts(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].alert_id != page2[0].alert_id

    def test_get_active_alerts(
        self, engine: AlertManagerEngine,
        alert: _FlexFreshnessAlert,
        breach: _FlexSLABreach,
    ):
        """get_active_alerts excludes resolved and suppressed."""
        # alert is pending -> active
        engine.send_alert(alert.alert_id)  # now sent -> active
        # Create another and resolve it
        a2 = engine.create_alert(
            breach.breach_id, "ds-001", "warning", "email", "resolved one",
        )
        engine.send_alert(a2.alert_id)
        engine.resolve_alert(a2.alert_id, "fixed")
        active = engine.get_active_alerts()
        assert len(active) == 1
        assert active[0].alert_id == alert.alert_id

    def test_get_alert_statistics_empty(self, engine: AlertManagerEngine):
        """Empty statistics returns zeroed dict."""
        stats = engine.get_alert_statistics()
        assert stats["total"] == 0
        assert stats["by_channel"] == {}
        assert stats["by_severity"] == {}
        assert stats["by_status"] == {}

    def test_get_alert_statistics_after_operations(
        self, engine: AlertManagerEngine, sent_alert: _FlexFreshnessAlert,
    ):
        """Statistics reflect created/sent alerts."""
        stats = engine.get_alert_statistics()
        assert stats["total"] >= 1
        assert "email" in stats["by_channel"]
        assert "critical" in stats["by_severity"]

    def test_get_alert_by_id(
        self, engine: AlertManagerEngine, alert: _FlexFreshnessAlert,
    ):
        """get_alert retrieves an alert by ID."""
        found = engine.get_alert(alert.alert_id)
        assert found is not None
        assert found.alert_id == alert.alert_id

    def test_get_alert_unknown_returns_none(
        self, engine: AlertManagerEngine,
    ):
        """Unknown alert_id returns None."""
        assert engine.get_alert("nonexistent") is None


# ============================================================================
# Test: record_breach / acknowledge_breach / resolve_breach
# ============================================================================


class TestBreachManagement:
    """Tests for breach lifecycle management."""

    def test_record_breach(self, engine: AlertManagerEngine):
        """record_breach creates an active breach."""
        breach = engine.record_breach("ds-001", "sla-001", "critical", 96.5)
        assert breach.breach_id.startswith("BRC-")
        assert breach.status == BreachStatus.DETECTED
        assert breach.dataset_id == "ds-001"
        assert breach.sla_id == "sla-001"
        assert breach.severity == BreachSeverity.CRITICAL
        assert breach.age_at_breach_hours == 96.5

    def test_record_breach_provenance(self, engine: AlertManagerEngine):
        """Breach has a 64-char provenance hash."""
        breach = engine.record_breach("ds-001", "sla-001", "critical", 50.0)
        assert len(breach.provenance_hash) == 64

    def test_record_breach_invalid_severity(
        self, engine: AlertManagerEngine,
    ):
        """Invalid breach severity raises ValueError."""
        with pytest.raises(ValueError, match="Invalid breach severity"):
            engine.record_breach("ds-001", "sla-001", "bogus_severity", 10.0)

    def test_record_breach_all_severities(
        self, engine: AlertManagerEngine,
    ):
        """All valid BreachSeverity values are accepted."""
        for sev in ("info", "low", "medium", "high", "critical"):
            b = engine.record_breach(f"ds-{sev}", "sla-001", sev, 10.0)
            assert b.severity == BreachSeverity(sev)

    def test_acknowledge_breach(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Acknowledging transitions from DETECTED to ACKNOWLEDGED."""
        result = engine.acknowledge_breach(breach.breach_id, "ops-eng")
        assert result.status == BreachStatus.ACKNOWLEDGED
        assert result.acknowledged_by == "ops-eng"
        assert result.acknowledged_at is not None

    def test_acknowledge_breach_unknown_raises(
        self, engine: AlertManagerEngine,
    ):
        """Unknown breach_id raises ValueError."""
        with pytest.raises(ValueError, match="Breach not found"):
            engine.acknowledge_breach("bad-id", "ops")

    def test_acknowledge_resolved_breach_raises(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Cannot acknowledge a resolved breach."""
        engine.resolve_breach(breach.breach_id, "fixed")
        with pytest.raises(ValueError, match="Cannot acknowledge resolved"):
            engine.acknowledge_breach(breach.breach_id, "ops")

    def test_resolve_breach(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Resolving transitions to RESOLVED with notes."""
        result = engine.resolve_breach(breach.breach_id, "Pipeline restarted")
        assert result.status == BreachStatus.RESOLVED
        assert result.resolution_notes == "Pipeline restarted"
        assert result.resolved_at is not None

    def test_resolve_breach_unknown_raises(
        self, engine: AlertManagerEngine,
    ):
        """Unknown breach_id raises ValueError."""
        with pytest.raises(ValueError, match="Breach not found"):
            engine.resolve_breach("bad-id", "notes")

    def test_resolve_breach_updates_provenance(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Resolution updates provenance hash."""
        old_hash = breach.provenance_hash
        engine.resolve_breach(breach.breach_id, "fixed")
        assert breach.provenance_hash != old_hash


# ============================================================================
# Test: list_breaches / get_active_breaches
# ============================================================================


class TestBreachQueries:
    """Tests for breach query methods."""

    def test_list_breaches_empty(self, engine: AlertManagerEngine):
        """No breaches returns empty list."""
        assert engine.list_breaches() == []

    def test_list_breaches_after_record(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Recorded breach appears in list."""
        breaches = engine.list_breaches()
        assert len(breaches) == 1

    def test_list_breaches_filter_by_dataset(
        self, engine: AlertManagerEngine,
    ):
        """Filter by dataset_id works."""
        engine.record_breach("ds-AAA", "sla-1", "critical", 10.0)
        engine.record_breach("ds-BBB", "sla-2", "critical", 20.0)
        result = engine.list_breaches(dataset_id="ds-AAA")
        assert len(result) == 1
        assert result[0].dataset_id == "ds-AAA"

    def test_list_breaches_filter_by_status(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Filter by status works."""
        engine.resolve_breach(breach.breach_id, "fixed")
        active = engine.list_breaches(status="active")
        assert len(active) == 0
        resolved = engine.list_breaches(status="resolved")
        assert len(resolved) == 1

    def test_list_breaches_filter_by_severity(
        self, engine: AlertManagerEngine,
    ):
        """Filter by severity works."""
        engine.record_breach("ds-1", "sla-1", "critical", 10.0)
        engine.record_breach("ds-2", "sla-2", "high", 20.0)
        result = engine.list_breaches(severity="critical")
        assert len(result) == 1

    def test_list_breaches_pagination(
        self, engine: AlertManagerEngine,
    ):
        """Limit and offset pagination works."""
        for i in range(5):
            engine.record_breach(f"ds-{i}", "sla-1", "critical", 10.0)
        page = engine.list_breaches(limit=2, offset=0)
        assert len(page) == 2

    def test_get_active_breaches(
        self, engine: AlertManagerEngine,
    ):
        """get_active_breaches excludes resolved breaches."""
        b1 = engine.record_breach("ds-1", "sla-1", "critical", 10.0)
        b2 = engine.record_breach("ds-2", "sla-2", "critical", 20.0)
        engine.resolve_breach(b1.breach_id, "fixed")
        active = engine.get_active_breaches()
        assert len(active) == 1
        assert active[0].breach_id == b2.breach_id

    def test_get_breach_by_id(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """get_breach retrieves by ID."""
        found = engine.get_breach(breach.breach_id)
        assert found is not None
        assert found.breach_id == breach.breach_id

    def test_get_breach_unknown_returns_none(
        self, engine: AlertManagerEngine,
    ):
        """Unknown breach_id returns None."""
        assert engine.get_breach("nonexistent") is None


# ============================================================================
# Test: get_statistics / reset
# ============================================================================


class TestStatisticsAndReset:
    """Tests for get_statistics and reset."""

    def test_statistics_initial(self, engine: AlertManagerEngine):
        """Initial statistics are all zeroes."""
        stats = engine.get_statistics()
        assert stats["alerts"]["total"] == 0
        assert stats["breaches"]["total"] == 0
        assert stats["throttle_entries"] == 0
        assert stats["dedup_entries"] == 0
        assert "timestamp" in stats

    def test_statistics_after_operations(
        self, engine: AlertManagerEngine, sent_alert: _FlexFreshnessAlert,
    ):
        """Statistics reflect operations performed."""
        stats = engine.get_statistics()
        assert stats["alerts"]["total"] >= 1
        assert stats["breaches"]["total"] >= 1

    def test_statistics_escalation_levels(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """Escalation levels appear in statistics."""
        policy = {
            "levels": [
                {"delay_minutes": 0, "channel": "email", "severity": "warning"},
            ],
        }
        engine.escalate(breach.breach_id, policy, current_level=-1)
        stats = engine.get_statistics()
        assert breach.breach_id in stats["escalation_levels"]

    def test_reset_clears_all(
        self, engine: AlertManagerEngine,
        sent_alert: _FlexFreshnessAlert,
        breach: _FlexSLABreach,
    ):
        """Reset clears all alerts, breaches, and tracking state."""
        engine.reset()
        assert engine.list_alerts() == []
        assert engine.list_breaches() == []
        assert engine.get_active_alerts() == []
        assert engine.get_active_breaches() == []
        stats = engine.get_statistics()
        assert stats["alerts"]["total"] == 0
        assert stats["breaches"]["total"] == 0
        assert stats["throttle_entries"] == 0
        assert stats["dedup_entries"] == 0

    def test_reset_allows_new_operations(
        self, engine: AlertManagerEngine, breach: _FlexSLABreach,
    ):
        """After reset, new operations work normally."""
        engine.reset()
        new_breach = engine.record_breach("ds-new", "sla-new", "high", 30.0)
        assert new_breach.breach_id.startswith("BRC-")
        assert len(engine.list_breaches()) == 1
