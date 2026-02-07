# -*- coding: utf-8 -*-
"""
Unit tests for AlertingConfig (OBS-004)

Tests configuration defaults, environment variable parsing, singleton
behavior, and channel-specific configuration.

Coverage target: 85%+ of config.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from greenlang.infrastructure.alerting_service.config import (
    AlertingConfig,
    get_config,
    set_config,
    reset_config,
    _DEFAULT_SEVERITY_ROUTING,
)


# ============================================================================
# AlertingConfig default values
# ============================================================================


class TestAlertingConfig:
    """Test suite for AlertingConfig dataclass."""

    def test_default_values(self):
        """Test that all default configuration values are correct."""
        config = AlertingConfig()

        assert config.service_name == "greenlang-alerting"
        assert config.environment == "production"
        assert config.enabled is True
        assert config.pagerduty_enabled is False
        assert config.opsgenie_enabled is False
        assert config.slack_enabled is False
        assert config.email_enabled is False
        assert config.teams_enabled is False
        assert config.webhook_enabled is False
        assert config.escalation_enabled is True
        assert config.analytics_enabled is True

    def test_env_var_loading(self, monkeypatch):
        """Test GL_ALERTING_* env vars override defaults."""
        monkeypatch.setenv("GL_ALERTING_SERVICE_NAME", "env-alerting")
        monkeypatch.setenv("GL_ALERTING_ENVIRONMENT", "staging")
        monkeypatch.setenv("GL_ALERTING_ENABLED", "false")

        config = AlertingConfig.from_env()

        assert config.service_name == "env-alerting"
        assert config.environment == "staging"
        assert config.enabled is False

    def test_pagerduty_config(self, monkeypatch):
        """Test PagerDuty fields are populated from env vars."""
        monkeypatch.setenv("GL_ALERTING_PAGERDUTY_ENABLED", "true")
        monkeypatch.setenv("GL_ALERTING_PAGERDUTY_ROUTING_KEY", "R_KEY_123")
        monkeypatch.setenv("GL_ALERTING_PAGERDUTY_API_KEY", "A_KEY_456")
        monkeypatch.setenv("GL_ALERTING_PAGERDUTY_SERVICE_ID", "P_SVC_789")

        config = AlertingConfig.from_env()

        assert config.pagerduty_enabled is True
        assert config.pagerduty_routing_key == "R_KEY_123"
        assert config.pagerduty_api_key == "A_KEY_456"
        assert config.pagerduty_service_id == "P_SVC_789"

    def test_opsgenie_config(self, monkeypatch):
        """Test Opsgenie fields are populated from env vars."""
        monkeypatch.setenv("GL_ALERTING_OPSGENIE_ENABLED", "true")
        monkeypatch.setenv("GL_ALERTING_OPSGENIE_API_KEY", "OG_KEY_001")
        monkeypatch.setenv("GL_ALERTING_OPSGENIE_API_URL", "https://custom.opsgenie.com")
        monkeypatch.setenv("GL_ALERTING_OPSGENIE_TEAM", "infra")

        config = AlertingConfig.from_env()

        assert config.opsgenie_enabled is True
        assert config.opsgenie_api_key == "OG_KEY_001"
        assert config.opsgenie_api_url == "https://custom.opsgenie.com"
        assert config.opsgenie_team == "infra"

    def test_slack_config(self, monkeypatch):
        """Test Slack 3-webhook URLs are populated from env vars."""
        monkeypatch.setenv("GL_ALERTING_SLACK_ENABLED", "true")
        monkeypatch.setenv("GL_ALERTING_SLACK_WEBHOOK_CRITICAL", "https://hooks.slack.com/crit")
        monkeypatch.setenv("GL_ALERTING_SLACK_WEBHOOK_WARNING", "https://hooks.slack.com/warn")
        monkeypatch.setenv("GL_ALERTING_SLACK_WEBHOOK_INFO", "https://hooks.slack.com/info")

        config = AlertingConfig.from_env()

        assert config.slack_enabled is True
        assert config.slack_webhook_critical == "https://hooks.slack.com/crit"
        assert config.slack_webhook_warning == "https://hooks.slack.com/warn"
        assert config.slack_webhook_info == "https://hooks.slack.com/info"

    def test_email_config(self, monkeypatch):
        """Test email SES vs SMTP mode configuration."""
        monkeypatch.setenv("GL_ALERTING_EMAIL_ENABLED", "true")
        monkeypatch.setenv("GL_ALERTING_EMAIL_FROM", "noreply@test.io")
        monkeypatch.setenv("GL_ALERTING_EMAIL_USE_SES", "false")
        monkeypatch.setenv("GL_ALERTING_EMAIL_SMTP_HOST", "smtp.test.io")
        monkeypatch.setenv("GL_ALERTING_EMAIL_SMTP_PORT", "465")
        monkeypatch.setenv("GL_ALERTING_EMAIL_SES_REGION", "us-east-1")

        config = AlertingConfig.from_env()

        assert config.email_enabled is True
        assert config.email_from == "noreply@test.io"
        assert config.email_use_ses is False
        assert config.email_smtp_host == "smtp.test.io"
        assert config.email_smtp_port == 465
        assert config.email_ses_region == "us-east-1"

    def test_teams_config(self):
        """Test Teams is disabled by default."""
        config = AlertingConfig()

        assert config.teams_enabled is False
        assert config.teams_webhook_url == ""

    def test_webhook_config(self):
        """Test generic webhook is disabled by default."""
        config = AlertingConfig()

        assert config.webhook_enabled is False
        assert config.webhook_url == ""
        assert config.webhook_secret == ""

    def test_severity_routing_defaults(self):
        """Test critical->PD+OG+Slack, warning->Slack+Email, info->Email."""
        config = AlertingConfig()

        assert config.default_severity_routing["critical"] == [
            "pagerduty", "opsgenie", "slack",
        ]
        assert config.default_severity_routing["warning"] == ["slack", "email"]
        assert config.default_severity_routing["info"] == ["email"]

    def test_escalation_defaults(self):
        """Test 15min ack timeout and 24h resolve timeout."""
        config = AlertingConfig()

        assert config.escalation_ack_timeout_minutes == 15
        assert config.escalation_resolve_timeout_hours == 24

    def test_dedup_window(self):
        """Test default dedup window is 60 minutes."""
        config = AlertingConfig()

        assert config.dedup_window_minutes == 60

    def test_rate_limits(self):
        """Test 120/min global and 60/min per channel rate limits."""
        config = AlertingConfig()

        assert config.rate_limit_per_minute == 120
        assert config.rate_limit_per_channel_per_minute == 60

    def test_analytics_retention(self):
        """Test default analytics retention is 365 days."""
        config = AlertingConfig()

        assert config.analytics_retention_days == 365

    def test_get_config_singleton(self, monkeypatch):
        """Test get_config() returns the same instance on repeated calls."""
        # Clear env so defaults apply
        for key in list(os.environ):
            if key.startswith("GL_ALERTING_"):
                monkeypatch.delenv(key, raising=False)

        reset_config()
        first = get_config()
        second = get_config()

        assert first is second

    def test_disabled_config(self):
        """Test enabled=False disables the alerting service."""
        config = AlertingConfig(enabled=False)

        assert config.enabled is False

    def test_set_config_replaces_singleton(self):
        """Test set_config replaces the singleton instance."""
        custom = AlertingConfig(service_name="custom-alerting")
        set_config(custom)

        retrieved = get_config()
        assert retrieved.service_name == "custom-alerting"

    def test_from_env_invalid_int_uses_default(self, monkeypatch):
        """Test invalid integer env var falls back to default."""
        monkeypatch.setenv("GL_ALERTING_DEDUP_WINDOW_MINUTES", "not_a_number")

        config = AlertingConfig.from_env()

        assert config.dedup_window_minutes == 60

    def test_severity_routing_is_independent_dict(self):
        """Test severity routing dicts are independent top-level copies.

        The default_factory uses ``dict(_DEFAULT_SEVERITY_ROUTING)`` which
        creates a shallow copy -- each config gets its own dict object, so
        assigning a whole new key on one does not affect the other.
        """
        config1 = AlertingConfig()
        config2 = AlertingConfig()

        config1.default_severity_routing["emergency"] = ["pagerduty"]

        assert "emergency" not in config2.default_severity_routing
