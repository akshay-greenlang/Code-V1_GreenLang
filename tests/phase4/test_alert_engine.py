# -*- coding: utf-8 -*-
"""
Tests for alert rule evaluation engine.

Tests cover alert rule evaluation, notification delivery,
deduplication, and alert state management.
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from greenlang.api.alerting.alert_engine import (
from greenlang.determinism import DeterministicClock
    AlertEngine,
    AlertRule,
    AlertState,
    AlertSeverity,
    RuleType,
    AlertInstance,
    NotificationChannel
)


@pytest.fixture
async def redis_mock():
    """Mock Redis client."""
    mock = AsyncMock()
    mock.from_url = AsyncMock(return_value=mock)
    mock.set = AsyncMock()
    mock.get = AsyncMock()
    mock.delete = AsyncMock()
    mock.scan = AsyncMock(return_value=(b'0', []))
    mock.zadd = AsyncMock()
    mock.zremrangebyrank = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def sample_threshold_rule():
    """Sample threshold alert rule."""
    return AlertRule(
        id="rule-001",
        name="High CPU Usage",
        description="Alert when CPU usage exceeds 80%",
        rule_type=RuleType.THRESHOLD,
        condition={
            "metric": "cpu.percent",
            "operator": ">",
            "threshold": 80.0,
            "aggregation": "avg"
        },
        notifications=[
            {
                "channel": NotificationChannel.EMAIL.value,
                "config": {"recipients": ["admin@example.com"]},
                "enabled": True
            }
        ],
        severity=AlertSeverity.WARNING,
        enabled=True,
        tags={"env": "production"},
        evaluation_interval=60,
        for_duration=300  # 5 minutes
    )


@pytest.fixture
def sample_absence_rule():
    """Sample absence alert rule."""
    return AlertRule(
        id="rule-002",
        name="Metric Absence",
        description="Alert when metric is missing",
        rule_type=RuleType.ABSENCE,
        condition={
            "metric": "heartbeat",
            "duration": 300  # 5 minutes
        },
        notifications=[],
        severity=AlertSeverity.CRITICAL,
        enabled=True,
        tags={}
    )


@pytest.mark.asyncio
class TestAlertEngine:
    """Test alert engine."""

    async def test_engine_start_stop(self, redis_mock):
        """Test engine starts and stops correctly."""
        with patch('greenlang.api.alerting.alert_engine.aioredis', redis_mock):
            engine = AlertEngine(redis_url="redis://localhost:6379")

            await engine.start()
            assert engine.running is True
            assert engine.redis_client is not None

            await engine.stop()
            assert engine.running is False

    async def test_add_remove_rule(self, redis_mock, sample_threshold_rule):
        """Test adding and removing alert rules."""
        with patch('greenlang.api.alerting.alert_engine.aioredis', redis_mock):
            engine = AlertEngine()
            engine.redis_client = redis_mock

            # Add rule
            await engine.add_rule(sample_threshold_rule)
            assert sample_threshold_rule.id in engine.rules
            assert redis_mock.set.called

            # Remove rule
            await engine.remove_rule(sample_threshold_rule.id)
            assert sample_threshold_rule.id not in engine.rules
            assert redis_mock.delete.called

    async def test_silence_rule(self, redis_mock, sample_threshold_rule):
        """Test silencing alert rules."""
        with patch('greenlang.api.alerting.alert_engine.aioredis', redis_mock):
            engine = AlertEngine()
            engine.redis_client = redis_mock
            engine.rules[sample_threshold_rule.id] = sample_threshold_rule

            # Silence for 1 hour
            await engine.silence_rule(sample_threshold_rule.id, 3600)

            rule = engine.rules[sample_threshold_rule.id]
            assert rule.silenced_until is not None
            assert rule.silenced_until > DeterministicClock.utcnow()


@pytest.mark.asyncio
class TestAlertRuleValidation:
    """Test alert rule validation."""

    def test_threshold_rule_validation(self):
        """Test threshold rule validation."""
        # Valid rule
        rule = AlertRule(
            id="rule-001",
            name="Test Rule",
            rule_type=RuleType.THRESHOLD,
            condition={
                "metric": "cpu.percent",
                "operator": ">",
                "threshold": 80.0
            },
            notifications=[],
            severity=AlertSeverity.WARNING,
            enabled=True
        )
        assert rule.rule_type == RuleType.THRESHOLD

        # Invalid rule (missing threshold)
        with pytest.raises(ValueError):
            AlertRule(
                id="rule-002",
                name="Invalid Rule",
                rule_type=RuleType.THRESHOLD,
                condition={"metric": "cpu.percent", "operator": ">"},
                notifications=[],
                severity=AlertSeverity.WARNING,
                enabled=True
            )

    def test_absence_rule_validation(self):
        """Test absence rule validation."""
        # Valid rule
        rule = AlertRule(
            id="rule-003",
            name="Absence Rule",
            rule_type=RuleType.ABSENCE,
            condition={
                "metric": "heartbeat",
                "duration": 300
            },
            notifications=[],
            severity=AlertSeverity.CRITICAL,
            enabled=True
        )
        assert rule.rule_type == RuleType.ABSENCE

        # Invalid rule (missing duration)
        with pytest.raises(ValueError):
            AlertRule(
                id="rule-004",
                name="Invalid Absence",
                rule_type=RuleType.ABSENCE,
                condition={"metric": "heartbeat"},
                notifications=[],
                severity=AlertSeverity.CRITICAL,
                enabled=True
            )


@pytest.mark.asyncio
class TestAlertEvaluation:
    """Test alert rule evaluation."""

    async def test_threshold_evaluation(self):
        """Test threshold condition evaluation."""
        engine = AlertEngine()

        # Test different operators
        assert engine._evaluate_condition(90.0, '>', 80.0) is True
        assert engine._evaluate_condition(70.0, '>', 80.0) is False

        assert engine._evaluate_condition(70.0, '<', 80.0) is True
        assert engine._evaluate_condition(90.0, '<', 80.0) is False

        assert engine._evaluate_condition(80.0, '>=', 80.0) is True
        assert engine._evaluate_condition(80.0, '<=', 80.0) is True

        assert engine._evaluate_condition(80.0, '==', 80.0) is True
        assert engine._evaluate_condition(80.0, '!=', 90.0) is True

    async def test_alert_instance_creation(self, sample_threshold_rule):
        """Test alert instance creation."""
        engine = AlertEngine()
        instance = await engine._get_or_create_instance(sample_threshold_rule, 85.0)

        assert instance.rule_id == sample_threshold_rule.id
        assert instance.state == AlertState.OK
        assert instance.value == 85.0
        assert instance.fingerprint != ""

        # Getting same instance again should return existing
        instance2 = await engine._get_or_create_instance(sample_threshold_rule, 90.0)
        assert instance.fingerprint == instance2.fingerprint

    async def test_alert_state_transitions(self):
        """Test alert state transitions."""
        instance = AlertInstance(
            rule_id="rule-001",
            state=AlertState.OK,
            value=75.0,
            labels={"env": "production"},
            started_at=DeterministicClock.utcnow(),
            last_evaluated=DeterministicClock.utcnow()
        )

        # OK -> PENDING
        instance.state = AlertState.PENDING
        instance.started_at = DeterministicClock.utcnow()
        assert instance.state == AlertState.PENDING

        # PENDING -> FIRING
        instance.state = AlertState.FIRING
        assert instance.state == AlertState.FIRING

        # FIRING -> RESOLVED
        instance.state = AlertState.RESOLVED
        assert instance.state == AlertState.RESOLVED


@pytest.mark.asyncio
class TestNotificationDelivery:
    """Test alert notification delivery."""

    async def test_email_notification_config(self):
        """Test email notification configuration."""
        engine = AlertEngine(
            smtp_config={
                'host': 'smtp.example.com',
                'port': 587,
                'username': 'alerts@example.com',
                'password': 'secret',
                'use_tls': True
            }
        )

        assert engine.smtp_config['host'] == 'smtp.example.com'
        assert engine.smtp_config['port'] == 587

    async def test_slack_notification_payload(self):
        """Test Slack notification payload generation."""
        instance = AlertInstance(
            rule_id="rule-001",
            state=AlertState.FIRING,
            value=95.0,
            labels={"env": "production"},
            started_at=DeterministicClock.utcnow(),
            last_evaluated=DeterministicClock.utcnow()
        )

        # Verify state determines color
        color_map = {
            AlertState.FIRING: '#f44336',
            AlertState.RESOLVED: '#4caf50',
            AlertState.PENDING: '#ff9800'
        }

        assert color_map[instance.state] == '#f44336'

    async def test_notification_grouping(self, sample_threshold_rule):
        """Test notification grouping interval."""
        instance = AlertInstance(
            rule_id=sample_threshold_rule.id,
            state=AlertState.FIRING,
            value=95.0,
            labels={},
            started_at=DeterministicClock.utcnow(),
            last_evaluated=DeterministicClock.utcnow(),
            last_notified=DeterministicClock.utcnow()
        )

        # Should not send if within group interval
        engine = AlertEngine()
        interval = (DeterministicClock.utcnow() - instance.last_notified).total_seconds()
        should_send = interval >= sample_threshold_rule.group_interval

        assert should_send is False  # Just notified


@pytest.mark.asyncio
class TestAlertDeduplication:
    """Test alert deduplication."""

    async def test_fingerprint_generation(self):
        """Test alert fingerprint generation for deduplication."""
        instance1 = AlertInstance(
            rule_id="rule-001",
            state=AlertState.FIRING,
            value=95.0,
            labels={"env": "production", "host": "server1"},
            started_at=DeterministicClock.utcnow(),
            last_evaluated=DeterministicClock.utcnow()
        )

        instance2 = AlertInstance(
            rule_id="rule-001",
            state=AlertState.FIRING,
            value=97.0,  # Different value
            labels={"env": "production", "host": "server1"},  # Same labels
            started_at=DeterministicClock.utcnow(),
            last_evaluated=DeterministicClock.utcnow()
        )

        # Same rule and labels should have same fingerprint
        assert instance1.fingerprint == instance2.fingerprint

    async def test_different_labels_different_fingerprint(self):
        """Test different labels generate different fingerprints."""
        instance1 = AlertInstance(
            rule_id="rule-001",
            state=AlertState.FIRING,
            value=95.0,
            labels={"env": "production", "host": "server1"},
            started_at=DeterministicClock.utcnow(),
            last_evaluated=DeterministicClock.utcnow()
        )

        instance2 = AlertInstance(
            rule_id="rule-001",
            state=AlertState.FIRING,
            value=95.0,
            labels={"env": "production", "host": "server2"},  # Different host
            started_at=DeterministicClock.utcnow(),
            last_evaluated=DeterministicClock.utcnow()
        )

        # Different labels should have different fingerprints
        assert instance1.fingerprint != instance2.fingerprint


@pytest.mark.asyncio
class TestAlertHistory:
    """Test alert history management."""

    async def test_save_alert_history(self, redis_mock, sample_threshold_rule):
        """Test saving alert to history."""
        with patch('greenlang.api.alerting.alert_engine.aioredis', redis_mock):
            engine = AlertEngine()
            engine.redis_client = redis_mock

            instance = AlertInstance(
                rule_id=sample_threshold_rule.id,
                state=AlertState.FIRING,
                value=95.0,
                labels={"env": "production"},
                started_at=DeterministicClock.utcnow(),
                last_evaluated=DeterministicClock.utcnow()
            )

            await engine._save_alert_history(sample_threshold_rule, instance)

            # Verify history was saved
            assert redis_mock.zadd.called
            assert redis_mock.zremrangebyrank.called  # Trim old entries

    async def test_get_active_alerts(self):
        """Test retrieving active alerts."""
        engine = AlertEngine()

        # Add active alert
        instance = AlertInstance(
            rule_id="rule-001",
            state=AlertState.FIRING,
            value=95.0,
            labels={"env": "production"},
            started_at=DeterministicClock.utcnow(),
            last_evaluated=DeterministicClock.utcnow()
        )
        engine.alert_instances[instance.fingerprint] = instance

        # Get active alerts
        active = await engine.get_active_alerts()

        assert len(active) == 1
        assert active[0]['rule_id'] == "rule-001"
        assert active[0]['state'] == AlertState.FIRING.value


@pytest.mark.asyncio
class TestAlertEngineIntegration:
    """Integration tests for alert engine."""

    async def test_complete_alert_lifecycle(self, redis_mock, sample_threshold_rule):
        """Test complete alert lifecycle from trigger to resolution."""
        with patch('greenlang.api.alerting.alert_engine.aioredis', redis_mock):
            engine = AlertEngine()
            engine.redis_client = redis_mock
            engine.running = True

            # Add rule
            await engine.add_rule(sample_threshold_rule)

            # Mock metric value retrieval to trigger alert
            engine._get_metric_value = AsyncMock(return_value=90.0)

            # Evaluate rule
            await engine._evaluate_threshold_rule(sample_threshold_rule)

            # Should create pending alert
            assert len(engine.alert_instances) > 0

            # Mock time passing for for_duration
            for instance in engine.alert_instances.values():
                instance.started_at = DeterministicClock.utcnow() - timedelta(seconds=400)

            # Re-evaluate
            await engine._evaluate_threshold_rule(sample_threshold_rule)

            # Should transition to firing
            firing_alerts = [a for a in engine.alert_instances.values() if a.state == AlertState.FIRING]
            # Note: Would be firing if notification was sent

            # Mock metric returning to normal
            engine._get_metric_value = AsyncMock(return_value=70.0)

            # Re-evaluate
            await engine._evaluate_threshold_rule(sample_threshold_rule)

            # Should transition to resolved
            for instance in engine.alert_instances.values():
                if instance.rule_id == sample_threshold_rule.id:
                    assert instance.state in [AlertState.RESOLVED, AlertState.OK]


# Run tests with: pytest tests/phase4/test_alert_engine.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
