# -*- coding: utf-8 -*-
"""
Unit tests for MonitoringAlertEngine - AGENT-EUDR-017 Engine 7

Tests continuous supplier risk monitoring and alert generation with
configurable frequency, multi-severity alerting, sanction screening,
watchlist management, and portfolio risk aggregation.

Target: 50+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-017 Supplier Risk Scorer (GL-EUDR-SRS-017)
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import pytest

from greenlang.agents.eudr.supplier_risk_scorer.monitoring_alert_engine import (
    MonitoringAlertEngine,
)
from greenlang.agents.eudr.supplier_risk_scorer.models import (
    AlertSeverity,
    AlertType,
    MonitoringFrequency,
    RiskLevel,
)


class TestMonitoringAlertEngineInit:
    """Tests for MonitoringAlertEngine initialization."""

    @pytest.mark.unit
    def test_initialization(self, mock_config):
        engine = MonitoringAlertEngine()
        assert engine._monitoring_configs == {}
        assert engine._alerts == {}
        assert engine._watchlist == set()


class TestConfigureMonitoring:
    """Tests for configure_monitoring method."""

    @pytest.mark.unit
    def test_configure_monitoring_creates_config(
        self, monitoring_alert_engine
    ):
        result = monitoring_alert_engine.configure_monitoring(
            supplier_id="SUPP-001",
            frequency=MonitoringFrequency.WEEKLY,
            alert_thresholds={
                "info": 25,
                "warning": 50,
                "high": 75,
                "critical": 90,
            },
        )
        assert result is not None
        assert result["frequency"] == MonitoringFrequency.WEEKLY

    @pytest.mark.unit
    @pytest.mark.parametrize("frequency", [
        MonitoringFrequency.DAILY,
        MonitoringFrequency.WEEKLY,
        MonitoringFrequency.BIWEEKLY,
        MonitoringFrequency.MONTHLY,
        MonitoringFrequency.QUARTERLY,
    ])
    def test_configure_all_frequencies(
        self, monitoring_alert_engine, frequency
    ):
        """Test all 5 monitoring frequencies."""
        result = monitoring_alert_engine.configure_monitoring(
            supplier_id=f"SUPP-{frequency.value}",
            frequency=frequency,
        )
        assert result["frequency"] == frequency


class TestCheckAlerts:
    """Tests for check_alerts method."""

    @pytest.mark.unit
    def test_check_alerts_detects_threshold_breach(
        self, monitoring_alert_engine
    ):
        # Configure monitoring first
        monitoring_alert_engine.configure_monitoring(
            supplier_id="SUPP-ALERT",
            frequency=MonitoringFrequency.DAILY,
        )
        # Check with high risk score
        alerts = monitoring_alert_engine.check_alerts(
            supplier_id="SUPP-ALERT",
            current_risk_score=Decimal("85.0"),
        )
        assert len(alerts) > 0


class TestGenerateAlertAllTypes:
    """Tests for generating all alert types."""

    @pytest.mark.unit
    @pytest.mark.parametrize("alert_type", [
        AlertType.RISK_THRESHOLD,
        AlertType.CERTIFICATION_EXPIRY,
        AlertType.DOCUMENT_MISSING,
        AlertType.DD_OVERDUE,
        AlertType.SANCTION_HIT,
        AlertType.BEHAVIOR_CHANGE,
    ])
    def test_generate_alert_all_types(
        self, monitoring_alert_engine, alert_type
    ):
        """Test generation of all 6 alert types."""
        alert = monitoring_alert_engine.generate_alert(
            supplier_id="SUPP-001",
            alert_type=alert_type,
            severity=AlertSeverity.WARNING,
            description=f"Test alert for {alert_type.value}",
        )
        assert alert["alert_type"] == alert_type


class TestScreenSanctions:
    """Tests for sanction list screening."""

    @pytest.mark.unit
    def test_screen_sanctions_clean(self, monitoring_alert_engine):
        result = monitoring_alert_engine.screen_sanctions(
            supplier_name="Clean Company Ltd",
            tax_id="BR123456789",
        )
        assert result["hit"] is False

    @pytest.mark.unit
    def test_screen_sanctions_hit(self, monitoring_alert_engine):
        # Use a known sanctioned name
        result = monitoring_alert_engine.screen_sanctions(
            supplier_name="COMPANY_SANCTIONED_1",
            tax_id="XX000000000",
        )
        assert result["hit"] is True


class TestWatchlistManagement:
    """Tests for watchlist management."""

    @pytest.mark.unit
    def test_add_to_watchlist(self, monitoring_alert_engine):
        result = monitoring_alert_engine.add_to_watchlist(
            supplier_id="SUPP-WATCH",
            reason="High risk score",
        )
        assert result["added"] is True

    @pytest.mark.unit
    def test_remove_from_watchlist(self, monitoring_alert_engine):
        # First add
        monitoring_alert_engine.add_to_watchlist(
            supplier_id="SUPP-WATCH2",
            reason="High risk",
        )
        # Then remove
        result = monitoring_alert_engine.remove_from_watchlist("SUPP-WATCH2")
        assert result["removed"] is True

    @pytest.mark.unit
    def test_watchlist_max_size(
        self, monitoring_alert_engine, mock_config
    ):
        # Try to add more than max
        for i in range(mock_config.watchlist_max_size + 10):
            monitoring_alert_engine.add_to_watchlist(
                supplier_id=f"SUPP-{i:04d}",
                reason="Test",
            )
        # Should be capped at max
        assert len(monitoring_alert_engine._watchlist) <= mock_config.watchlist_max_size


class TestScheduleReassessment:
    """Tests for automated reassessment scheduling."""

    @pytest.mark.unit
    def test_schedule_reassessment(self, monitoring_alert_engine):
        result = monitoring_alert_engine.schedule_reassessment(
            supplier_id="SUPP-REASS",
            trigger="risk_increase",
            next_date=datetime.now(timezone.utc) + timedelta(days=30),
        )
        assert "scheduled" in result
        assert result["scheduled"] is True


class TestHeatmap:
    """Tests for risk heat map generation."""

    @pytest.mark.unit
    def test_generate_risk_heatmap(self, monitoring_alert_engine):
        suppliers = [
            {"supplier_id": "SUPP-A", "risk_score": 30.0, "country": "BR"},
            {"supplier_id": "SUPP-B", "risk_score": 70.0, "country": "ID"},
            {"supplier_id": "SUPP-C", "risk_score": 50.0, "country": "MY"},
        ]
        heatmap = monitoring_alert_engine.generate_risk_heatmap(suppliers)
        assert "by_country" in heatmap
        assert "by_risk_level" in heatmap


class TestPortfolioRisk:
    """Tests for portfolio risk aggregation."""

    @pytest.mark.unit
    def test_aggregate_portfolio_risk(self, monitoring_alert_engine):
        suppliers = [
            {"supplier_id": "SUPP-A", "risk_score": 30.0},
            {"supplier_id": "SUPP-B", "risk_score": 70.0},
            {"supplier_id": "SUPP-C", "risk_score": 50.0},
        ]
        portfolio = monitoring_alert_engine.aggregate_portfolio_risk(suppliers)
        assert "average_risk" in portfolio
        assert "high_risk_count" in portfolio


class TestAcknowledgeAlert:
    """Tests for alert acknowledgment."""

    @pytest.mark.unit
    def test_acknowledge_alert(self, monitoring_alert_engine):
        # First generate alert
        alert = monitoring_alert_engine.generate_alert(
            supplier_id="SUPP-001",
            alert_type=AlertType.RISK_THRESHOLD,
            severity=AlertSeverity.HIGH,
            description="Test alert",
        )
        # Acknowledge it
        result = monitoring_alert_engine.acknowledge_alert(
            alert_id=alert["alert_id"],
            acknowledged_by="user@example.com",
            notes="Reviewed and addressed",
        )
        assert result["acknowledged"] is True


class TestAlertSeverities:
    """Tests for all alert severity levels."""

    @pytest.mark.unit
    @pytest.mark.parametrize("severity", [
        AlertSeverity.INFO,
        AlertSeverity.WARNING,
        AlertSeverity.HIGH,
        AlertSeverity.CRITICAL,
    ])
    def test_generate_alert_all_severities(
        self, monitoring_alert_engine, severity
    ):
        """Test all 4 severity levels."""
        alert = monitoring_alert_engine.generate_alert(
            supplier_id="SUPP-SEV",
            alert_type=AlertType.RISK_THRESHOLD,
            severity=severity,
            description=f"Test {severity.value} alert",
        )
        assert alert["severity"] == severity


class TestProvenance:
    """Tests for provenance tracking."""

    @pytest.mark.unit
    def test_alert_includes_provenance_hash(self, monitoring_alert_engine):
        alert = monitoring_alert_engine.generate_alert(
            supplier_id="SUPP-PROV",
            alert_type=AlertType.RISK_THRESHOLD,
            severity=AlertSeverity.WARNING,
            description="Test",
        )
        assert "provenance_hash" in alert


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.unit
    def test_invalid_supplier_id_raises_error(self, monitoring_alert_engine):
        with pytest.raises(ValueError):
            monitoring_alert_engine.configure_monitoring(
                supplier_id="",
                frequency=MonitoringFrequency.WEEKLY,
            )
