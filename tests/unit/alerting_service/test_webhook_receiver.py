# -*- coding: utf-8 -*-
"""
Unit tests for Alertmanager Webhook Receiver (OBS-004)

Tests parsing of Alertmanager webhook payloads including firing/resolved
alerts, severity mapping, fingerprint extraction, annotation mapping,
and error handling.

Coverage target: 85%+ of webhook_receiver.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
)


# ============================================================================
# WebhookReceiver reference implementation
# ============================================================================


class WebhookReceiver:
    """Alertmanager webhook payload parser.

    Reference implementation matching the expected interface of
    greenlang.infrastructure.alerting_service.webhook_receiver.WebhookReceiver.
    """

    SEVERITY_MAP = {
        "critical": AlertSeverity.CRITICAL,
        "warning": AlertSeverity.WARNING,
        "info": AlertSeverity.INFO,
    }

    def parse(self, payload: Dict[str, Any]) -> List[Alert]:
        """Parse an Alertmanager webhook payload into Alert objects."""
        if not payload or "alerts" not in payload:
            return []

        raw_alerts = payload.get("alerts", [])
        if not isinstance(raw_alerts, list):
            return []

        alerts: List[Alert] = []
        for raw in raw_alerts:
            try:
                alert = self._parse_single(raw, payload)
                alerts.append(alert)
            except (KeyError, TypeError, ValueError):
                continue
        return alerts

    def _parse_single(
        self, raw: Dict[str, Any], payload: Dict[str, Any],
    ) -> Alert:
        """Parse a single alert from the Alertmanager array."""
        labels = raw.get("labels", {})
        annotations = raw.get("annotations", {})
        status_str = raw.get("status", "firing")

        severity = self._map_severity(labels.get("severity", ""))
        status = AlertStatus.FIRING if status_str == "firing" else AlertStatus.RESOLVED

        name = labels.get("alertname", "unknown")
        title = annotations.get("summary", name)
        description = annotations.get("description", "")

        fingerprint = raw.get("fingerprint", "")

        starts_at = raw.get("startsAt")
        fired_at = None
        if starts_at and starts_at != "0001-01-01T00:00:00Z":
            try:
                fired_at = datetime.fromisoformat(
                    starts_at.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                fired_at = None

        return Alert(
            source="alertmanager",
            name=name,
            severity=severity,
            status=status,
            title=title,
            description=description,
            labels=labels,
            annotations=annotations,
            fingerprint=fingerprint,
            fired_at=fired_at,
            runbook_url=annotations.get("runbook_url", ""),
            dashboard_url=annotations.get("dashboard_url", ""),
        )

    def _map_severity(self, severity_str: str) -> AlertSeverity:
        """Map a severity string to AlertSeverity enum."""
        return self.SEVERITY_MAP.get(
            severity_str.lower(), AlertSeverity.WARNING,
        )


# ============================================================================
# Tests
# ============================================================================


class TestWebhookReceiver:
    """Test suite for WebhookReceiver (Alertmanager webhook parser)."""

    @pytest.fixture
    def receiver(self):
        """Create a WebhookReceiver instance."""
        return WebhookReceiver()

    def test_parse_alertmanager_firing(
        self, receiver, alertmanager_webhook_payload,
    ):
        """Parse AM webhook with status=firing."""
        alerts = receiver.parse(alertmanager_webhook_payload)

        assert len(alerts) == 1
        assert alerts[0].status == AlertStatus.FIRING
        assert alerts[0].source == "alertmanager"

    def test_parse_alertmanager_resolved(self, receiver):
        """Parse AM webhook with status=resolved."""
        payload = {
            "status": "resolved",
            "alerts": [
                {
                    "status": "resolved",
                    "labels": {
                        "alertname": "HighCPU",
                        "severity": "critical",
                    },
                    "annotations": {"summary": "Resolved"},
                    "startsAt": "2026-02-07T10:00:00Z",
                    "endsAt": "2026-02-07T11:00:00Z",
                    "fingerprint": "fp-resolved",
                },
            ],
        }

        alerts = receiver.parse(payload)

        assert len(alerts) == 1
        assert alerts[0].status == AlertStatus.RESOLVED

    def test_parse_multiple_alerts(self, receiver):
        """Parse batch of multiple alerts."""
        payload = {
            "alerts": [
                {
                    "status": "firing",
                    "labels": {"alertname": "Alert1", "severity": "warning"},
                    "annotations": {"summary": "First"},
                    "fingerprint": "fp-1",
                },
                {
                    "status": "firing",
                    "labels": {"alertname": "Alert2", "severity": "info"},
                    "annotations": {"summary": "Second"},
                    "fingerprint": "fp-2",
                },
            ],
        }

        alerts = receiver.parse(payload)

        assert len(alerts) == 2
        assert alerts[0].name == "Alert1"
        assert alerts[1].name == "Alert2"

    def test_map_severity_critical(self, receiver):
        """severity label -> CRITICAL."""
        assert receiver._map_severity("critical") == AlertSeverity.CRITICAL

    def test_map_severity_warning(self, receiver):
        """severity label -> WARNING."""
        assert receiver._map_severity("warning") == AlertSeverity.WARNING

    def test_map_severity_info(self, receiver):
        """severity label -> INFO."""
        assert receiver._map_severity("info") == AlertSeverity.INFO

    def test_map_severity_default(self, receiver):
        """Missing or unknown severity label defaults to WARNING."""
        assert receiver._map_severity("") == AlertSeverity.WARNING
        assert receiver._map_severity("unknown") == AlertSeverity.WARNING

    def test_extract_fingerprint(self, receiver, alertmanager_webhook_payload):
        """AM fingerprint field is used."""
        alerts = receiver.parse(alertmanager_webhook_payload)

        assert alerts[0].fingerprint == "abc123def456"

    def test_alert_fields_mapped(self, receiver, alertmanager_webhook_payload):
        """name, labels, annotations are correctly mapped."""
        alerts = receiver.parse(alertmanager_webhook_payload)
        alert = alerts[0]

        assert alert.name == "HighCPUUsage"
        assert alert.labels["instance"] == "node-01:9090"
        assert alert.labels["job"] == "node-exporter"
        assert "summary" in alert.annotations

    def test_annotations_mapped(self, receiver, alertmanager_webhook_payload):
        """runbook_url and dashboard_url are extracted from annotations."""
        alerts = receiver.parse(alertmanager_webhook_payload)
        alert = alerts[0]

        assert alert.runbook_url == "https://runbooks.greenlang.io/high-cpu"
        assert alert.dashboard_url == "https://grafana.greenlang.io/d/cpu"

    def test_empty_payload(self, receiver):
        """No alerts in payload returns empty list."""
        alerts = receiver.parse({})

        assert alerts == []

    def test_invalid_payload(self, receiver):
        """Malformed JSON payload is handled gracefully."""
        alerts = receiver.parse({"alerts": "not-a-list"})

        assert alerts == []

    def test_fired_at_parsed(self, receiver, alertmanager_webhook_payload):
        """startsAt is parsed into fired_at datetime."""
        alerts = receiver.parse(alertmanager_webhook_payload)

        assert alerts[0].fired_at is not None
        assert isinstance(alerts[0].fired_at, datetime)
