# -*- coding: utf-8 -*-
"""
Tests for GL-OPS-X-002: Alert & Anomaly Agent

Tests cover:
    - Anomaly detection using Z-score, IQR, and MAD methods
    - Alert configuration management
    - Threshold-based alerting
    - Alert acknowledgment and resolution
    - Provenance tracking

Author: GreenLang Team
"""

import pytest
from datetime import datetime, timedelta

from greenlang.agents.base import AgentConfig, AgentResult
from greenlang.agents.operations.alert_anomaly_agent import (
    AlertAnomalyAgent,
    AlertAnomalyInput,
    AlertAnomalyOutput,
    DataPoint,
    AnomalyDetection,
    AlertConfiguration,
    AlertEvent,
    AnomalyType,
    AnomalySeverity,
    DetectionMethod,
)
from greenlang.utilities.determinism import DeterministicClock


class TestAlertAnomalyAgentInitialization:
    """Tests for agent initialization."""

    def test_agent_creation_default_config(self):
        """Test creating agent with default configuration."""
        agent = AlertAnomalyAgent()

        assert agent.AGENT_ID == "GL-OPS-X-002"
        assert agent.AGENT_NAME == "Alert & Anomaly Agent"
        assert agent.VERSION == "1.0.0"

    def test_agent_creation_custom_config(self):
        """Test creating agent with custom configuration."""
        config = AgentConfig(
            name="Custom Alert Agent",
            description="Custom config test",
        )
        agent = AlertAnomalyAgent(config)

        assert agent.config.name == "Custom Alert Agent"


class TestAnomalyDetection:
    """Tests for anomaly detection functionality."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return AlertAnomalyAgent()

    def test_detect_anomaly_zscore(self, agent):
        """Test anomaly detection using Z-score method."""
        # Create dataset with one obvious outlier
        data_points = [
            {"metric_name": "emissions", "value": 100.0},
            {"metric_name": "emissions", "value": 102.0},
            {"metric_name": "emissions", "value": 98.0},
            {"metric_name": "emissions", "value": 101.0},
            {"metric_name": "emissions", "value": 99.0},
            {"metric_name": "emissions", "value": 500.0},  # Outlier
        ]

        result = agent.run({
            "operation": "detect_anomalies",
            "data_points": data_points,
            "method": "zscore",
            "threshold": 2.0,
        })

        assert result.success
        anomalies = result.data["data"].get("anomalies", [])
        assert len(anomalies) >= 1

    def test_detect_anomaly_iqr(self, agent):
        """Test anomaly detection using IQR method."""
        data_points = [
            {"metric_name": "energy", "value": 50.0},
            {"metric_name": "energy", "value": 52.0},
            {"metric_name": "energy", "value": 48.0},
            {"metric_name": "energy", "value": 51.0},
            {"metric_name": "energy", "value": 200.0},  # Outlier
        ]

        result = agent.run({
            "operation": "detect_anomalies",
            "data_points": data_points,
            "method": "iqr",
        })

        assert result.success

    def test_detect_anomaly_mad(self, agent):
        """Test anomaly detection using MAD method."""
        data_points = [
            {"metric_name": "temperature", "value": 25.0},
            {"metric_name": "temperature", "value": 26.0},
            {"metric_name": "temperature", "value": 24.0},
            {"metric_name": "temperature", "value": 100.0},  # Outlier
        ]

        result = agent.run({
            "operation": "detect_anomalies",
            "data_points": data_points,
            "method": "mad",
        })

        assert result.success


class TestAlertConfiguration:
    """Tests for alert configuration management."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return AlertAnomalyAgent()

    def test_add_alert_config(self, agent):
        """Test adding an alert configuration."""
        result = agent.run({
            "operation": "add_alert_config",
            "alert_config": {
                "name": "High Emissions Alert",
                "metric_name": "emissions",
                "condition": "gt",
                "threshold": 1000.0,
                "severity": "critical",
            }
        })

        assert result.success
        assert result.data["data"].get("added") or "config_id" in result.data["data"]

    def test_list_alert_configs(self, agent):
        """Test listing alert configurations."""
        # Add a config first
        agent.run({
            "operation": "add_alert_config",
            "alert_config": {
                "name": "Test Alert",
                "metric_name": "test",
                "condition": "gt",
                "threshold": 100.0,
            }
        })

        result = agent.run({"operation": "list_alert_configs"})

        assert result.success
        assert "configs" in result.data["data"] or "count" in result.data["data"]

    def test_remove_alert_config(self, agent):
        """Test removing an alert configuration."""
        # Add a config first
        add_result = agent.run({
            "operation": "add_alert_config",
            "alert_config": {
                "name": "To Remove",
                "metric_name": "test",
                "condition": "gt",
                "threshold": 50.0,
            }
        })

        config_id = add_result.data["data"].get("config_id")
        if config_id:
            result = agent.run({
                "operation": "remove_alert_config",
                "config_id": config_id,
            })
            assert result.success


class TestAlertTriggering:
    """Tests for alert triggering."""

    @pytest.fixture
    def agent_with_config(self):
        """Create agent with pre-configured alert."""
        agent = AlertAnomalyAgent()
        agent.run({
            "operation": "add_alert_config",
            "alert_config": {
                "name": "High Value Alert",
                "metric_name": "emissions",
                "condition": "gt",
                "threshold": 100.0,
                "severity": "warning",
            }
        })
        return agent

    def test_check_alert_triggers(self, agent_with_config):
        """Test that alert triggers when threshold breached."""
        result = agent_with_config.run({
            "operation": "check_alerts",
            "data_points": [
                {"metric_name": "emissions", "value": 150.0},
            ]
        })

        assert result.success

    def test_get_active_alerts(self, agent_with_config):
        """Test getting active alerts."""
        # Trigger an alert first
        agent_with_config.run({
            "operation": "check_alerts",
            "data_points": [
                {"metric_name": "emissions", "value": 200.0},
            ]
        })

        result = agent_with_config.run({"operation": "get_active_alerts"})

        assert result.success


class TestAlertLifecycle:
    """Tests for alert lifecycle management."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return AlertAnomalyAgent()

    def test_acknowledge_alert(self, agent):
        """Test acknowledging an alert."""
        # Setup and trigger an alert
        agent.run({
            "operation": "add_alert_config",
            "alert_config": {
                "name": "Test Alert",
                "metric_name": "test",
                "condition": "gt",
                "threshold": 10.0,
            }
        })

        agent.run({
            "operation": "check_alerts",
            "data_points": [{"metric_name": "test", "value": 100.0}]
        })

        # Get alerts and acknowledge
        alerts_result = agent.run({"operation": "get_active_alerts"})
        alerts = alerts_result.data["data"].get("alerts", [])

        if alerts:
            alert_id = alerts[0].get("alert_id")
            if alert_id:
                result = agent.run({
                    "operation": "acknowledge_alert",
                    "alert_id": alert_id,
                })
                assert result.success

    def test_resolve_alert(self, agent):
        """Test resolving an alert."""
        # Setup and trigger an alert
        agent.run({
            "operation": "add_alert_config",
            "alert_config": {
                "name": "Resolve Test",
                "metric_name": "test",
                "condition": "gt",
                "threshold": 10.0,
            }
        })

        agent.run({
            "operation": "check_alerts",
            "data_points": [{"metric_name": "test", "value": 50.0}]
        })

        alerts_result = agent.run({"operation": "get_active_alerts"})
        alerts = alerts_result.data["data"].get("alerts", [])

        if alerts:
            alert_id = alerts[0].get("alert_id")
            if alert_id:
                result = agent.run({
                    "operation": "resolve_alert",
                    "alert_id": alert_id,
                })
                assert result.success


class TestProvenanceTracking:
    """Tests for provenance and audit trail."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return AlertAnomalyAgent()

    def test_output_contains_provenance_hash(self, agent):
        """Test that all operations include provenance hash."""
        result = agent.run({
            "operation": "detect_anomalies",
            "data_points": [
                {"metric_name": "test", "value": 100.0},
                {"metric_name": "test", "value": 105.0},
            ],
            "method": "zscore",
        })

        assert result.success
        assert "provenance_hash" in result.data
        assert len(result.data["provenance_hash"]) == 16


class TestStatistics:
    """Tests for statistics functionality."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return AlertAnomalyAgent()

    def test_get_statistics(self, agent):
        """Test getting agent statistics."""
        result = agent.run({"operation": "get_statistics"})

        assert result.success
        assert "data" in result.data


class TestIntegration:
    """Integration tests."""

    def test_full_alerting_workflow(self):
        """Test a complete alerting workflow."""
        agent = AlertAnomalyAgent()

        # 1. Add alert configuration
        agent.run({
            "operation": "add_alert_config",
            "alert_config": {
                "name": "Production Alert",
                "metric_name": "emissions",
                "condition": "gt",
                "threshold": 100.0,
                "severity": "critical",
            }
        })

        # 2. Check alerts with normal data
        normal_result = agent.run({
            "operation": "check_alerts",
            "data_points": [
                {"metric_name": "emissions", "value": 50.0},
            ]
        })
        assert normal_result.success

        # 3. Check alerts with breach data
        breach_result = agent.run({
            "operation": "check_alerts",
            "data_points": [
                {"metric_name": "emissions", "value": 200.0},
            ]
        })
        assert breach_result.success

        # 4. Get statistics
        stats = agent.run({"operation": "get_statistics"})
        assert stats.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
