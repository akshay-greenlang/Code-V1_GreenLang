# -*- coding: utf-8 -*-
"""
Unit tests for SLOServiceConfig (OBS-005)

Tests configuration defaults, environment variable parsing, singleton
behavior, burn rate windows, error budget thresholds, and budget
exhaustion policies.

Coverage target: 85%+ of config.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from greenlang.infrastructure.slo_service.config import (
    SLOServiceConfig,
    get_config,
    set_config,
    reset_config,
    _DEFAULT_BURN_RATE_WINDOWS,
)


class TestSLOServiceConfig:
    """Test suite for SLOServiceConfig dataclass."""

    def test_default_config_values(self):
        """All default values are correct out of the box."""
        config = SLOServiceConfig()

        assert config.service_name == "greenlang-slo"
        assert config.environment == "production"
        assert config.enabled is True
        assert config.prometheus_url == "http://prometheus:9090"
        assert config.prometheus_timeout_seconds == 30
        assert config.prometheus_verify_ssl is True
        assert config.grafana_url == "http://grafana:3000"
        assert config.grafana_org_id == 1
        assert config.redis_url == "redis://redis:6379/3"
        assert config.redis_cache_ttl_seconds == 60
        assert config.database_pool_size == 5
        assert config.evaluation_interval_seconds == 60
        assert config.evaluation_batch_size == 50
        assert config.compliance_enabled is True
        assert config.alerting_bridge_enabled is True

    def test_env_var_overrides(self, monkeypatch):
        """GL_SLO_* environment variables override defaults."""
        monkeypatch.setenv("GL_SLO_SERVICE_NAME", "custom-slo")
        monkeypatch.setenv("GL_SLO_ENVIRONMENT", "staging")
        monkeypatch.setenv("GL_SLO_ENABLED", "false")
        monkeypatch.setenv("GL_SLO_PROMETHEUS_URL", "http://prom:9999")

        config = SLOServiceConfig.from_env()

        assert config.service_name == "custom-slo"
        assert config.environment == "staging"
        assert config.enabled is False
        assert config.prometheus_url == "http://prom:9999"

    def test_burn_rate_window_defaults(self):
        """Burn rate windows have correct default thresholds."""
        config = SLOServiceConfig()

        fast = config.burn_rate_windows["fast"]
        assert fast["threshold"] == 14.4
        assert fast["long_window"] == "1h"
        assert fast["short_window"] == "5m"

        medium = config.burn_rate_windows["medium"]
        assert medium["threshold"] == 6.0
        assert medium["long_window"] == "6h"
        assert medium["short_window"] == "30m"

        slow = config.burn_rate_windows["slow"]
        assert slow["threshold"] == 1.0
        assert slow["long_window"] == "3d"
        assert slow["short_window"] == "6h"

    def test_error_budget_thresholds(self):
        """Budget thresholds have correct default percentages."""
        config = SLOServiceConfig()

        assert config.budget_threshold_warning == 20.0
        assert config.budget_threshold_critical == 50.0
        assert config.budget_threshold_exhausted == 100.0

    def test_singleton_pattern(self, monkeypatch):
        """get_config returns the same instance on repeated calls."""
        for key in list(os.environ):
            if key.startswith("GL_SLO_"):
                monkeypatch.delenv(key, raising=False)

        reset_config()
        first = get_config()
        second = get_config()

        assert first is second

    def test_reset_config(self, monkeypatch):
        """reset_config clears the singleton so next call creates fresh."""
        for key in list(os.environ):
            if key.startswith("GL_SLO_"):
                monkeypatch.delenv(key, raising=False)

        reset_config()
        first = get_config()
        reset_config()
        second = get_config()

        assert first is not second

    def test_config_validation_bool_true_variants(self, monkeypatch):
        """Boolean env vars accept 'true', '1', 'yes'."""
        for val in ("true", "1", "yes", "TRUE", "Yes"):
            monkeypatch.setenv("GL_SLO_ENABLED", val)
            config = SLOServiceConfig.from_env()
            assert config.enabled is True, f"Failed for value: {val}"

    def test_config_validation_bool_false_variants(self, monkeypatch):
        """Boolean env vars treat anything else as False."""
        for val in ("false", "0", "no", "nope"):
            monkeypatch.setenv("GL_SLO_ENABLED", val)
            config = SLOServiceConfig.from_env()
            assert config.enabled is False, f"Failed for value: {val}"

    def test_prometheus_url_from_env(self, monkeypatch):
        """Prometheus URL can be overridden via env."""
        monkeypatch.setenv("GL_SLO_PROMETHEUS_URL", "http://prom.svc:9090")
        config = SLOServiceConfig.from_env()
        assert config.prometheus_url == "http://prom.svc:9090"

    def test_redis_url_default(self):
        """Redis URL defaults to redis://redis:6379/3."""
        config = SLOServiceConfig()
        assert config.redis_url == "redis://redis:6379/3"

    def test_database_url_from_env(self, monkeypatch):
        """Database URL can be overridden via env."""
        monkeypatch.setenv("GL_SLO_DATABASE_URL", "postgresql://prod:pass@db:5432/slo")
        config = SLOServiceConfig.from_env()
        assert config.database_url == "postgresql://prod:pass@db:5432/slo"

    def test_reporting_settings(self):
        """Compliance reporting settings have correct defaults."""
        config = SLOServiceConfig()
        assert config.compliance_enabled is True
        assert config.compliance_weekly_day == "monday"
        assert config.compliance_retention_days == 365

    @pytest.mark.parametrize("action", ["freeze_deployments", "alert_only", "none"])
    def test_budget_exhausted_actions(self, action, monkeypatch):
        """Budget exhausted action accepts all valid values."""
        monkeypatch.setenv("GL_SLO_BUDGET_EXHAUSTED_ACTION", action)
        config = SLOServiceConfig.from_env()
        assert config.budget_exhausted_action == action

    def test_api_rate_limit_defaults(self):
        """Evaluation batch size acts as implicit rate limit."""
        config = SLOServiceConfig()
        assert config.evaluation_batch_size == 50

    def test_slo_definitions_path(self):
        """SLO definitions YAML path has correct default."""
        config = SLOServiceConfig()
        assert config.slo_definitions_path == "slo_definitions.yaml"

    def test_cache_ttl(self):
        """Redis cache TTL defaults to 60 seconds."""
        config = SLOServiceConfig()
        assert config.redis_cache_ttl_seconds == 60

    def test_set_config_replaces_singleton(self):
        """set_config replaces the singleton instance."""
        custom = SLOServiceConfig(service_name="replaced-slo")
        set_config(custom)

        retrieved = get_config()
        assert retrieved.service_name == "replaced-slo"

    def test_from_env_invalid_int_uses_default(self, monkeypatch):
        """Invalid integer env var falls back to default."""
        monkeypatch.setenv("GL_SLO_PROMETHEUS_TIMEOUT_SECONDS", "not_a_number")
        config = SLOServiceConfig.from_env()
        assert config.prometheus_timeout_seconds == 30

    def test_from_env_invalid_float_uses_default(self, monkeypatch):
        """Invalid float env var falls back to default."""
        monkeypatch.setenv("GL_SLO_BUDGET_THRESHOLD_WARNING", "bad_float")
        config = SLOServiceConfig.from_env()
        assert config.budget_threshold_warning == 20.0

    def test_burn_rate_windows_independent_copies(self):
        """Burn rate window dicts are independent between instances."""
        config1 = SLOServiceConfig()
        config2 = SLOServiceConfig()
        config1.burn_rate_windows["extreme"] = {"threshold": 50.0}
        assert "extreme" not in config2.burn_rate_windows
