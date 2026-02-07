# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Alerting Service Unit Tests (OBS-004)
=========================================================

Provides common fixtures for testing the Unified Alerting &
Notification Platform.  All external dependencies (httpx, boto3, SMTP)
are mocked so tests run without network access.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.alerting_service.config import (
    AlertingConfig,
    reset_config,
)
from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
    EscalationPolicy,
    EscalationStep,
    NotificationChannel,
    NotificationResult,
    NotificationStatus,
    OnCallUser,
)


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config():
    """Create an AlertingConfig with test defaults."""
    return AlertingConfig(
        service_name="test-alerting",
        environment="test",
        enabled=True,
        pagerduty_enabled=True,
        pagerduty_routing_key="test-pd-routing-key",
        pagerduty_api_key="test-pd-api-key",
        pagerduty_service_id="P_SVC_001",
        opsgenie_enabled=True,
        opsgenie_api_key="test-og-api-key",
        opsgenie_api_url="https://api.opsgenie.com",
        opsgenie_team="platform",
        slack_enabled=True,
        slack_webhook_critical="https://hooks.slack.com/critical",
        slack_webhook_warning="https://hooks.slack.com/warning",
        slack_webhook_info="https://hooks.slack.com/info",
        email_enabled=True,
        email_from="alerts@test.greenlang.io",
        email_use_ses=True,
        email_ses_region="eu-west-1",
        teams_enabled=False,
        teams_webhook_url="",
        webhook_enabled=False,
        webhook_url="",
        webhook_secret="",
        escalation_ack_timeout_minutes=15,
        escalation_resolve_timeout_hours=24,
        dedup_window_minutes=60,
        rate_limit_per_minute=120,
        rate_limit_per_channel_per_minute=60,
        analytics_retention_days=365,
    )


# ---------------------------------------------------------------------------
# Alert fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_alert():
    """Create a sample CRITICAL alert in FIRING state."""
    return Alert(
        source="prometheus",
        name="HighCPUUsage",
        severity=AlertSeverity.CRITICAL,
        title="CPU usage above 95% on node-01",
        description="Node node-01 has sustained CPU usage above 95% for 5 minutes.",
        labels={"instance": "node-01:9090", "job": "node-exporter"},
        annotations={
            "summary": "High CPU on node-01",
            "runbook_url": "https://runbooks.greenlang.io/high-cpu",
            "dashboard_url": "https://grafana.greenlang.io/d/cpu",
        },
        tenant_id="t-acme",
        team="platform",
        service="api-service",
        environment="production",
        runbook_url="https://runbooks.greenlang.io/high-cpu",
        dashboard_url="https://grafana.greenlang.io/d/cpu",
        related_trace_id="abc123def456",
    )


@pytest.fixture
def sample_alert_warning():
    """Create a sample WARNING alert in FIRING state."""
    return Alert(
        source="prometheus",
        name="DiskSpaceLow",
        severity=AlertSeverity.WARNING,
        title="Disk usage above 80% on /data",
        description="Volume /data is at 82% capacity.",
        labels={"instance": "db-01:9090", "mountpoint": "/data"},
        annotations={"summary": "Low disk space on db-01"},
        tenant_id="t-acme",
        team="data-platform",
        service="postgres",
        environment="production",
    )


@pytest.fixture
def sample_alert_info():
    """Create a sample INFO alert in FIRING state."""
    return Alert(
        source="loki",
        name="HighLogVolume",
        severity=AlertSeverity.INFO,
        title="Log volume exceeds 10K lines/min",
        description="api-service is producing high log volume.",
        labels={"service": "api-service", "namespace": "greenlang-prod"},
        tenant_id="t-acme",
        team="platform",
        service="api-service",
        environment="production",
    )


# ---------------------------------------------------------------------------
# Notification result fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_notification_result():
    """Create a successful NotificationResult."""
    return NotificationResult(
        channel=NotificationChannel.SLACK,
        status=NotificationStatus.SENT,
        recipient="https://hooks.slack.com/critical",
        duration_ms=123.4,
        response_code=200,
    )


# ---------------------------------------------------------------------------
# Escalation fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_escalation_policy():
    """Create an EscalationPolicy with 3 steps."""
    return EscalationPolicy(
        name="critical_default",
        steps=[
            EscalationStep(
                delay_minutes=0,
                channels=["pagerduty", "slack"],
                oncall_schedule_id="sched-001",
            ),
            EscalationStep(
                delay_minutes=15,
                channels=["pagerduty", "opsgenie", "slack"],
                notify_users=["mgr-001"],
            ),
            EscalationStep(
                delay_minutes=30,
                channels=["pagerduty", "opsgenie", "slack", "email"],
                notify_users=["mgr-001", "dir-001"],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# On-call fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_oncall_user():
    """Create a sample OnCallUser."""
    return OnCallUser(
        user_id="usr-pd-001",
        name="Jane Doe",
        email="jane.doe@greenlang.io",
        phone="+15551234567",
        provider="pagerduty",
        schedule_id="sched-001",
    )


# ---------------------------------------------------------------------------
# Mock HTTP client fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_httpx_client():
    """Create an AsyncMock of httpx.AsyncClient."""
    client = AsyncMock()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"status": "success"}
    response.text = '{"status": "success"}'
    response.raise_for_status = MagicMock()
    client.post.return_value = response
    client.get.return_value = response
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


# ---------------------------------------------------------------------------
# Alertmanager webhook payload fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def alertmanager_webhook_payload():
    """Create a dict matching the Alertmanager webhook JSON format."""
    return {
        "version": "4",
        "groupKey": "test-group-key",
        "truncatedAlerts": 0,
        "status": "firing",
        "receiver": "greenlang-webhook",
        "groupLabels": {"alertname": "HighCPUUsage"},
        "commonLabels": {
            "alertname": "HighCPUUsage",
            "severity": "critical",
            "job": "node-exporter",
        },
        "commonAnnotations": {
            "summary": "High CPU usage detected",
            "runbook_url": "https://runbooks.greenlang.io/high-cpu",
        },
        "externalURL": "http://alertmanager:9093",
        "alerts": [
            {
                "status": "firing",
                "labels": {
                    "alertname": "HighCPUUsage",
                    "severity": "critical",
                    "instance": "node-01:9090",
                    "job": "node-exporter",
                },
                "annotations": {
                    "summary": "High CPU usage on node-01",
                    "description": "CPU is at 96%.",
                    "runbook_url": "https://runbooks.greenlang.io/high-cpu",
                    "dashboard_url": "https://grafana.greenlang.io/d/cpu",
                },
                "startsAt": "2026-02-07T10:00:00.000Z",
                "endsAt": "0001-01-01T00:00:00Z",
                "generatorURL": "http://prometheus:9090/graph?g0.expr=...",
                "fingerprint": "abc123def456",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Singleton reset (autouse)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_alerting_config():
    """Reset the AlertingConfig singleton between tests."""
    yield
    reset_config()
