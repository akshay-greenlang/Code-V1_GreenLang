# -*- coding: utf-8 -*-
"""
Integration tests - Alertmanager Webhook (OBS-004)

Tests the Alertmanager webhook intake including single/batch alerts,
resolved status, label/annotation mapping, and error handling.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict
from unittest.mock import MagicMock

import pytest

from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
)

from tests.unit.alerting_service.test_webhook_receiver import WebhookReceiver


# ============================================================================
# Alertmanager webhook integration tests
# ============================================================================


class TestAlertmanagerWebhookIntegration:
    """Integration tests for Alertmanager webhook processing."""

    @pytest.fixture
    def receiver(self):
        return WebhookReceiver()

    def test_receive_single_firing_alert(self, receiver, alertmanager_payload):
        """Single firing alert is parsed correctly."""
        alerts = receiver.parse(alertmanager_payload)

        assert len(alerts) == 1
        assert alerts[0].status == AlertStatus.FIRING
        assert alerts[0].severity == AlertSeverity.CRITICAL
        assert alerts[0].name == "HighCPUUsage"

    def test_receive_batch_alerts(self, receiver):
        """Batch of 3 alerts is parsed correctly."""
        payload = {
            "alerts": [
                {
                    "status": "firing",
                    "labels": {"alertname": f"Alert{i}", "severity": "warning"},
                    "annotations": {"summary": f"Alert {i}"},
                    "fingerprint": f"fp-batch-{i}",
                }
                for i in range(3)
            ],
        }
        alerts = receiver.parse(payload)

        assert len(alerts) == 3
        for i, alert in enumerate(alerts):
            assert alert.name == f"Alert{i}"

    def test_receive_resolved_alert(self, receiver):
        """Resolved alert is parsed with RESOLVED status."""
        payload = {
            "alerts": [
                {
                    "status": "resolved",
                    "labels": {"alertname": "ResolvedAlert", "severity": "info"},
                    "annotations": {"summary": "All clear"},
                    "startsAt": "2026-02-07T10:00:00Z",
                    "endsAt": "2026-02-07T11:00:00Z",
                    "fingerprint": "fp-resolved-001",
                },
            ],
        }
        alerts = receiver.parse(payload)

        assert len(alerts) == 1
        assert alerts[0].status == AlertStatus.RESOLVED

    def test_receive_with_labels_and_annotations(self, receiver):
        """Labels and annotations are fully preserved."""
        payload = {
            "alerts": [
                {
                    "status": "firing",
                    "labels": {
                        "alertname": "AnnotatedAlert",
                        "severity": "critical",
                        "instance": "db-01:5432",
                        "job": "postgres",
                        "namespace": "greenlang-prod",
                    },
                    "annotations": {
                        "summary": "Database connection pool exhausted",
                        "description": "Pool at 100% for 5 minutes",
                        "runbook_url": "https://runbooks.greenlang.io/db-pool",
                        "dashboard_url": "https://grafana.greenlang.io/d/db",
                    },
                    "fingerprint": "fp-annotated",
                },
            ],
        }
        alerts = receiver.parse(payload)
        alert = alerts[0]

        assert alert.labels["instance"] == "db-01:5432"
        assert alert.labels["namespace"] == "greenlang-prod"
        assert alert.runbook_url == "https://runbooks.greenlang.io/db-pool"
        assert alert.dashboard_url == "https://grafana.greenlang.io/d/db"

    def test_receive_updates_existing_alert(self, receiver):
        """Two payloads with same fingerprint are parsed independently."""
        payload1 = {
            "alerts": [
                {
                    "status": "firing",
                    "labels": {"alertname": "SameAlert", "severity": "warning"},
                    "annotations": {"summary": "First"},
                    "fingerprint": "fp-update",
                },
            ],
        }
        payload2 = {
            "alerts": [
                {
                    "status": "resolved",
                    "labels": {"alertname": "SameAlert", "severity": "warning"},
                    "annotations": {"summary": "Resolved"},
                    "fingerprint": "fp-update",
                },
            ],
        }

        alerts1 = receiver.parse(payload1)
        alerts2 = receiver.parse(payload2)

        assert alerts1[0].status == AlertStatus.FIRING
        assert alerts2[0].status == AlertStatus.RESOLVED
        assert alerts1[0].fingerprint == alerts2[0].fingerprint

    def test_receive_invalid_payload_returns_empty(self, receiver):
        """Invalid payload returns empty list, no exception."""
        alerts = receiver.parse({"invalid": "data"})
        assert alerts == []

    def test_receive_with_external_url(self, receiver):
        """External URL in payload does not affect parsing."""
        payload = {
            "externalURL": "http://alertmanager:9093",
            "alerts": [
                {
                    "status": "firing",
                    "labels": {"alertname": "ExtURL", "severity": "info"},
                    "annotations": {"summary": "With external URL"},
                    "fingerprint": "fp-ext",
                    "generatorURL": "http://prometheus:9090/graph",
                },
            ],
        }
        alerts = receiver.parse(payload)

        assert len(alerts) == 1
        assert alerts[0].source == "alertmanager"

    def test_receive_with_group_labels(self, receiver):
        """Group labels in payload do not affect individual alert parsing."""
        payload = {
            "groupLabels": {"alertname": "GroupedAlert"},
            "commonLabels": {"severity": "warning", "team": "platform"},
            "alerts": [
                {
                    "status": "firing",
                    "labels": {"alertname": "GroupedAlert", "severity": "warning"},
                    "annotations": {"summary": "Grouped"},
                    "fingerprint": "fp-group",
                },
            ],
        }
        alerts = receiver.parse(payload)

        assert len(alerts) == 1
        assert alerts[0].name == "GroupedAlert"
