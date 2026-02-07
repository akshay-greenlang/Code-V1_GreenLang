# -*- coding: utf-8 -*-
"""
Unit tests for Alerting Bridge (OBS-005)

Tests SLO alert dispatch to OBS-004 unified alerting, including burn
rate alerts, budget alerts, alert resolution, and error handling.

Coverage target: 85%+ of alerting_bridge.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from greenlang.infrastructure.slo_service.alerting_bridge import AlertingBridge


class TestAlertingBridgeBurnRate:
    """Tests for burn rate alert dispatch."""

    def test_fire_slo_alert_fast_burn_critical(
        self, sample_burn_rate_alert, sample_slo
    ):
        """Fast burn alert dispatches with critical severity."""
        bridge = AlertingBridge(enabled=True)
        result = bridge.fire_burn_rate_alert(sample_burn_rate_alert, sample_slo)
        assert result["dispatched"] is True
        alert_data = result["alert_data"]
        assert alert_data["severity"] == "critical"
        assert alert_data["source"] == "slo-service"
        assert sample_slo.slo_id in alert_data["labels"]["slo_id"]

    def test_fire_slo_alert_medium_burn_warning(self, sample_slo):
        """Medium burn alert has warning severity."""
        from greenlang.infrastructure.slo_service.models import BurnRateAlert
        alert = BurnRateAlert(
            slo_id=sample_slo.slo_id,
            slo_name=sample_slo.name,
            burn_window="medium",
            burn_rate_long=7.0,
            burn_rate_short=6.5,
            threshold=6.0,
            severity="warning",
            service=sample_slo.service,
            message="Test medium burn",
        )
        bridge = AlertingBridge(enabled=True)
        result = bridge.fire_burn_rate_alert(alert, sample_slo)
        assert result["dispatched"] is True
        assert result["alert_data"]["severity"] == "warning"

    def test_fire_slo_alert_slow_burn_info(self, sample_slo):
        """Slow burn alert has info severity."""
        from greenlang.infrastructure.slo_service.models import BurnRateAlert
        alert = BurnRateAlert(
            slo_id=sample_slo.slo_id,
            burn_window="slow",
            burn_rate_long=1.5,
            burn_rate_short=1.2,
            threshold=1.0,
            severity="info",
            service=sample_slo.service,
            message="Test slow burn",
        )
        bridge = AlertingBridge(enabled=True)
        result = bridge.fire_burn_rate_alert(alert, sample_slo)
        assert result["dispatched"] is True
        assert result["alert_data"]["severity"] == "info"

    def test_fire_with_alerting_service(self, sample_burn_rate_alert, sample_slo):
        """Alert data is forwarded to the alerting service."""
        mock_svc = MagicMock()
        bridge = AlertingBridge(enabled=True, alerting_service=mock_svc)
        bridge.fire_burn_rate_alert(sample_burn_rate_alert, sample_slo)
        mock_svc.fire_alert.assert_called_once()


class TestAlertingBridgeBudget:
    """Tests for budget alert dispatch."""

    def test_fire_budget_alert_exhausted(self, sample_error_budget_exhausted, sample_slo):
        """Exhausted budget dispatches critical alert."""
        bridge = AlertingBridge(enabled=True)
        result = bridge.fire_budget_alert(sample_error_budget_exhausted, sample_slo)
        assert result["dispatched"] is True
        assert result["alert_data"]["severity"] == "critical"

    def test_fire_budget_alert_critical(self, sample_error_budget_critical, sample_slo):
        """Critical budget dispatches warning alert."""
        bridge = AlertingBridge(enabled=True)
        result = bridge.fire_budget_alert(sample_error_budget_critical, sample_slo)
        assert result["dispatched"] is True
        assert result["alert_data"]["severity"] == "warning"

    def test_alert_includes_slo_details(self, sample_error_budget, sample_slo):
        """Alert data includes SLO metadata."""
        bridge = AlertingBridge(enabled=True)
        result = bridge.fire_budget_alert(sample_error_budget, sample_slo)
        alert_data = result["alert_data"]
        assert alert_data["labels"]["slo_id"] == sample_slo.slo_id
        assert alert_data["labels"]["service"] == sample_slo.service

    def test_alert_includes_budget_info(self, sample_error_budget, sample_slo):
        """Alert annotations include budget remaining and SLI value."""
        bridge = AlertingBridge(enabled=True)
        result = bridge.fire_budget_alert(sample_error_budget, sample_slo)
        annotations = result["alert_data"]["annotations"]
        assert "budget_remaining_percent" in annotations
        assert "sli_value" in annotations


class TestAlertingBridgeResolve:
    """Tests for alert resolution."""

    def test_resolve_slo_alert(self, sample_slo):
        """Resolving an alert returns success."""
        bridge = AlertingBridge(enabled=True)
        result = bridge.resolve_alert(sample_slo, "burn_rate")
        assert result["resolved"] is True
        assert result["slo_id"] == sample_slo.slo_id


class TestAlertingBridgeDisabled:
    """Tests for disabled bridge."""

    def test_bridge_disabled_no_alerts(
        self, sample_burn_rate_alert, sample_slo
    ):
        """Disabled bridge does not dispatch alerts."""
        bridge = AlertingBridge(enabled=False)
        result = bridge.fire_burn_rate_alert(sample_burn_rate_alert, sample_slo)
        assert result["dispatched"] is False
        assert result["reason"] == "bridge_disabled"

    def test_bridge_disabled_no_budget_alerts(
        self, sample_error_budget, sample_slo
    ):
        """Disabled bridge does not dispatch budget alerts."""
        bridge = AlertingBridge(enabled=False)
        result = bridge.fire_budget_alert(sample_error_budget, sample_slo)
        assert result["dispatched"] is False

    def test_bridge_disabled_resolve(self, sample_slo):
        """Disabled bridge does not resolve alerts."""
        bridge = AlertingBridge(enabled=False)
        result = bridge.resolve_alert(sample_slo)
        assert result["resolved"] is False


class TestAlertingBridgeErrorHandling:
    """Tests for error handling in the bridge."""

    def test_bridge_error_handling(self, sample_burn_rate_alert, sample_slo):
        """Bridge handles alerting service errors gracefully."""
        mock_svc = MagicMock()
        mock_svc.fire_alert.side_effect = Exception("Service unavailable")
        bridge = AlertingBridge(enabled=True, alerting_service=mock_svc)
        result = bridge.fire_burn_rate_alert(sample_burn_rate_alert, sample_slo)
        assert result["dispatched"] is False
        assert "error" in result
