# -*- coding: utf-8 -*-
"""
Integration Test Fixtures for Alerting Service (OBS-004)
========================================================

Provides shared fixtures for integration testing the full alerting
pipeline: webhook intake -> parse -> fire -> deduplicate -> route ->
notify -> escalate -> resolve.

All external services (PagerDuty, Opsgenie, Slack, SES) are mocked
via httpx mock transport. When live endpoints are configured via env
vars, tests can optionally run against real services.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.alerting_service.config import (
    AlertingConfig,
    reset_config,
    set_config,
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
)


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

PAGERDUTY_ROUTING_KEY = os.getenv("GL_TEST_PAGERDUTY_ROUTING_KEY", "")
SKIP_LIVE = not PAGERDUTY_ROUTING_KEY

skip_without_live = pytest.mark.skipif(
    SKIP_LIVE,
    reason="GL_TEST_PAGERDUTY_ROUTING_KEY not set; skipping live integration tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def alerting_config():
    """AlertingConfig with test-friendly defaults and mock URLs."""
    config = AlertingConfig(
        service_name="integration-alerting",
        environment="test",
        enabled=True,
        pagerduty_enabled=True,
        pagerduty_routing_key="test-pd-key",
        pagerduty_api_key="test-pd-api-key",
        opsgenie_enabled=True,
        opsgenie_api_key="test-og-key",
        opsgenie_api_url="https://api.opsgenie.com",
        opsgenie_team="platform",
        slack_enabled=True,
        slack_webhook_critical="https://hooks.slack.com/test-critical",
        slack_webhook_warning="https://hooks.slack.com/test-warning",
        slack_webhook_info="https://hooks.slack.com/test-info",
        email_enabled=True,
        email_from="alerts@test.greenlang.io",
        email_use_ses=True,
        email_ses_region="eu-west-1",
        teams_enabled=True,
        teams_webhook_url="https://teams.webhook.test/incoming",
        webhook_enabled=True,
        webhook_url="https://webhook.test/alerts",
        webhook_secret="test-secret-key",
        dedup_window_minutes=60,
        escalation_ack_timeout_minutes=15,
        rate_limit_per_minute=120,
    )
    set_config(config)
    return config


@pytest.fixture
def mock_http_server():
    """Create an AsyncMock httpx client that simulates all channel APIs."""
    client = AsyncMock()

    # Default successful response
    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = {"status": "success"}
    success_response.text = '{"status": "success"}'
    success_response.raise_for_status = MagicMock()

    client.post.return_value = success_response
    client.get.return_value = success_response
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    return client


@pytest.fixture
def sample_critical_alert():
    """Create a CRITICAL alert for integration tests."""
    return Alert(
        source="prometheus",
        name="HighCPUUsage",
        severity=AlertSeverity.CRITICAL,
        title="CPU usage above 95% on node-01",
        description="Sustained high CPU for 5 minutes.",
        labels={"instance": "node-01:9090", "job": "node-exporter"},
        annotations={
            "summary": "High CPU on node-01",
            "runbook_url": "https://runbooks.greenlang.io/high-cpu",
            "dashboard_url": "https://grafana.greenlang.io/d/cpu",
        },
        tenant_id="t-integ-test",
        team="platform",
        service="api-service",
        environment="test",
        runbook_url="https://runbooks.greenlang.io/high-cpu",
        dashboard_url="https://grafana.greenlang.io/d/cpu",
    )


@pytest.fixture
def sample_warning_alert():
    """Create a WARNING alert for integration tests."""
    return Alert(
        source="prometheus",
        name="DiskSpaceLow",
        severity=AlertSeverity.WARNING,
        title="Disk usage above 80%",
        labels={"instance": "db-01:9090", "mountpoint": "/data"},
        tenant_id="t-integ-test",
        team="data-platform",
        service="postgres",
        environment="test",
    )


@pytest.fixture
def alertmanager_payload():
    """Full Alertmanager webhook payload for integration tests."""
    return {
        "version": "4",
        "groupKey": "integ-group-001",
        "status": "firing",
        "receiver": "greenlang-webhook",
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
                    "description": "CPU at 96%.",
                    "runbook_url": "https://runbooks.greenlang.io/high-cpu",
                },
                "startsAt": "2026-02-07T10:00:00.000Z",
                "endsAt": "0001-01-01T00:00:00Z",
                "fingerprint": "integ-fp-001",
            },
        ],
    }


@pytest.fixture(autouse=True)
def reset_config_after_test():
    """Reset config singleton after each integration test."""
    yield
    reset_config()


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest mock_agents to avoid unrelated agent patching.

    The parent tests/integration/conftest.py defines a mock_agents autouse
    fixture that patches GreenLang agent imports.  We override it here so
    alerting service tests are not affected by those patches.
    """
    yield


@pytest.fixture(scope="session", autouse=True)
def block_network():
    """Override parent conftest block_network.

    The parent tests/integration/conftest.py blocks all socket operations,
    which conflicts with pytest-asyncio event loop creation on Windows.
    Alerting service integration tests use mocked httpx clients and do
    not require network blocking.
    """
    yield
